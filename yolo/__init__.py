import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision

import os, pickle, random, time
import numpy as np
from PIL import Image

from .darknet import get_test_input, parse_cfg, create_modules, \
    EmptyLayer, DetectionLayer, DarkNet
from .util import unique, bbox_iou, predict_transform, write_results, \
    letterbox_image, prep_image, load_classes
# from .detect import *
# from .train import *

anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]],
                      np.float32) / 416

class MmwaveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size = 0, transforms = None):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir,x) for x in files]
        
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        
        if data_size == 0:
            data_size = len(files)
        
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.transforms = transforms
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = self.preprocess(image)
        
        labels_str = image_address.split("_") \
            [-1].split('[')[1].split(']')[0].split(',') # get the bb info from the filename
        labels = np.array([int(a) for a in labels_str]) # convert bb info to int array
        labels = np.append(labels, 1)

        image = image.astype(np.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, labels

    #Image preprocessing before feeding to network
    def preprocess(self, image):
        image = np.array(image.convert('RGB'))
        return image.transpose(2,1,0)
    

def get_absolute_yolo_box(y_pred, valid_anchors_wh, num_classes):
    """
    Given a cell offset prediction from the model, calculate the absolute box coordinates to the whole image.
    It's also an adpation of the original C code here:
    https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L83
    note that, we divide w and h by grid size 
    inputs:
    y_pred: Prediction tensor from the model output, in the shape of (batch, grid, grid, anchor, 5 + num_classes)
    outputs:
    y_box: boxes in shape of (batch, grid, grid, anchor, 4), the last dimension is (xmin, ymin, xmax, ymax)
    objectness: probability that an object exists
    classes: probability of classes
    """

    t_xy, t_wh, objectness, classes = tf.split(
        y_pred, (2, 2, 1, num_classes), axis=-1)

    objectness = tf.sigmoid(objectness)
    classes = tf.sigmoid(classes)

    grid_size = tf.shape(y_pred)[1]
    # meshgrid generates a grid that repeats by given range. It's the Cx and Cy in YoloV3 paper.
    # for example, tf.meshgrid(tf.range(3), tf.range(3)) will generate a list with two elements
    # note that in real code, the grid_size should be something like 13, 26, 52 for examples here and below
    #
    # [[0, 1, 2],
    #  [0, 1, 2],
    #  [0, 1, 2]]
    #
    # [[0, 0, 0],
    #  [1, 1, 1],
    #  [2, 2, 2]]
    #
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))

    # next, we stack two items in the list together in the last dimension, so that
    # we can interleve these elements together and become this:
    #
    # [[[0, 0], [1, 0], [2, 0]],
    #  [[0, 1], [1, 1], [2, 1]],
    #  [[0, 2], [1, 2], [2, 2]]]
    #
    C_xy = tf.stack(C_xy, axis=-1)

    # let's add an empty dimension at axis=2 to expand the tensor to this:
    #
    # [[[[0, 0]], [[1, 0]], [[2, 0]]],
    #  [[[0, 1]], [[1, 1]], [[2, 1]]],
    #  [[[0, 2]], [[1, 2]], [[2, 2]]]]
    #
    # at this moment, we now have a grid, which can always give us (y, x)
    # if we access grid[x][y]. For example, grid[0][1] == [[1, 0]]
    C_xy = tf.expand_dims(C_xy, axis=2)  # [gx, gy, 1, 2]

    # YoloV2, YoloV3:
    # bx = sigmoid(tx) + Cx
    # by = sigmoid(ty) + Cy
    #
    # for example, if all elements in b_xy are (0.1, 0.2), the result will be
    #
    # [[[[0.1, 0.2]], [[1.1, 0.2]], [[2.1, 0.2]]],
    #  [[[0.1, 1.2]], [[1.1, 1.2]], [[2.1, 1.2]]],
    #  [[[0.1, 2.2]], [[1.1, 2.2]], [[2.1, 2.2]]]]
    #
    b_xy = tf.sigmoid(t_xy) + tf.cast(C_xy, tf.float32)

    # finally, divide this absolute box_xy by grid_size, and then we will get the normalized bbox centroids
    # for each anchor in each grid cell. b_xy is now in shape (batch_size, grid_size, grid_size, num_anchor, 2)
    #
    # [[[[0.1/3, 0.2/3]], [[1.1/3, 0.2/3]], [[2.1/3, 0.2/3]]],
    #  [[[0.1/3, 1.2/3]], [[1.1/3, 1.2]/3], [[2.1/3, 1.2/3]]],
    #  [[[0.1/3, 2.2/3]], [[1.1/3, 2.2/3]], [[2.1/3, 2.2/3]]]]
    #
    b_xy = b_xy / tf.cast(grid_size, tf.float32)

    # YoloV2:
    # "If the cell is offset from the top left corner of the image by (cx , cy)
    # and the bounding box prior has width and height pw , ph , then the predictions correspond to: "
    #
    # https://github.com/pjreddie/darknet/issues/568#issuecomment-469600294
    # "It’s OK for the predicted box to be wider and/or taller than the original image, but
    # it does not make sense for the box to have a negative width or height. That’s why
    # we take the exponent of the predicted number."
    b_wh = tf.exp(t_wh) * valid_anchors_wh

    y_box = tf.concat([b_xy, b_wh], axis=-1)
    return y_box, objectness, classes


def get_relative_yolo_box(y_true, valid_anchors_wh):
    """
    This is the inverse of `get_absolute_yolo_box` above. It's turning (bx, by, bw, bh) into
    (tx, ty, tw, th) that is relative to cell location.
    """
    grid_size = tf.shape(y_true)[1]
    C_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    C_xy = tf.expand_dims(tf.stack(C_xy, axis=-1), axis=2)

    b_xy = y_true[..., 0:2]
    b_wh = y_true[..., 2:4]
    t_xy = b_xy * tf.cast(grid_size, tf.float32) - tf.cast(C_xy, tf.float32)

    t_wh = tf.math.log(b_wh / valid_anchors_wh)
    # b_wh could have some cells are 0, divided by anchor could result in inf or nan
    t_wh = tf.where(
        tf.logical_or(tf.math.is_inf(t_wh), tf.math.is_nan(t_wh)),
        tf.zeros_like(t_wh), t_wh)

    y_box = tf.concat([t_xy, t_wh], axis=-1)
    return y_box


class YOLOLoss(object):
    def __init__(self, num_classes, valid_anchors_wh):
        self.num_classes = num_classes
        self.ignore_thresh = 0.5
        self.valid_anchors_wh = valid_anchors_wh
        self.lambda_coord = 5.0
        self.lamda_noobj = 0.5

    def __call__(self, y_true, y_pred):
        """
        calculate the loss of model prediction for one scale
        """
        # for xy and wh, I seperated them into two groups with different suffix
        # suffix rel (relative) means that its coordinates are relative to cells
        # basically (tx, ty, tw, th) format from the paper
        # _rel is used to calcuate the loss
        # suffix abs (absolute) means that its coordinates are absolute with in whole image
        # basically (bx, by, bw, bh) format from the paper
        # _abs is used to calcuate iou and ignore mask

        # split y_pred into xy, wh, objectness and one-hot classes
        # pred_xy_rel: (batch, grid, grid, anchor, 2)
        # pred_wh_rel: (batch, grid, grid, anchor, 2)
        # TODO: Add comment for the sigmoid here
        pred_xy_rel = torch.sigmoid(y_pred[..., 0:2])
        pred_wh_rel = y_pred[..., 2:4]

        # this box is used to calculate iou, NOT loss. so we can't use
        # cell offset anymore and have to transform it into true values
        # both pred_obj and pred_class has been sigmoid'ed here
        # pred_xy_abs: (batch, grid, grid, anchor, 2)
        # pred_wh_abs: (batch, grid, grid, anchor, 2)
        # pred_obj: (batch, grid, grid, anchor, 1)
        # pred_class: (batch, grid, grid, anchor, num_classes)
        pred_box_abs, pred_obj, pred_class = get_absolute_yolo_box(
            y_pred, self.valid_anchors_wh, self.num_classes)
        pred_box_abs = xywh_to_x1x2y1y2(pred_box_abs)

        # split y_true into xy, wh, objectness and one-hot classes
        # pred_xy_abs: (batch, grid, grid, anchor, 2)
        # pred_wh_abs: (batch, grid, grid, anchor, 2)
        # pred_obj: (batch, grid, grid, anchor, 1)
        # pred_class: (batch, grid, grid, anchor, num_classes)
        true_xy_abs, true_wh_abs, true_obj, true_class = torch.split(
            y_true, (2, 2, 1, self.num_classes), dim=-1)
        true_box_abs = torch.cat([true_xy_abs, true_wh_abs], dim=-1)
        true_box_abs = xywh_to_x1x2y1y2(true_box_abs)

        # true_box_rel: (batch, grid, grid, anchor, 4)
        true_box_rel = get_relative_yolo_box(y_true, self.valid_anchors_wh)
        true_xy_rel = true_box_rel[..., 0:2]
        true_wh_rel = true_box_rel[..., 2:4]

        # some adjustment to improve small box detection, note the (2-truth.w*truth.h) below
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L190
        weight = 2 - true_wh_abs[..., 0] * true_wh_abs[..., 1]

        # YoloV2:
        # "If the cell is offset from the top left corner of the image by (cx , cy)
        # and the bounding box prior has width and height pw , ph , then the predictions correspond to:"
        #
        # to calculate the iou and determine the ignore mask, we need to first transform
        # prediction into real coordinates (bx, by, bw, bh)

        # YoloV2:
        # "This ground truth value can be easily computed by inverting the equations above."
        #
        # to calculate loss and differentiation, we need to transform ground truth into
        # cell offset first like demonstrated here:
        # https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c#L93
        xy_loss = self.calc_xy_loss(true_obj, true_xy_rel, pred_xy_rel, weight)
        wh_loss = self.calc_wh_loss(true_obj, true_wh_rel, pred_wh_rel, weight)
        class_loss = self.calc_class_loss(true_obj, true_class, pred_class)

        # use the absolute yolo box to calculate iou and ignore mask
        ignore_mask = self.calc_ignore_mask(true_obj, true_box_abs,
                                            pred_box_abs)
        obj_loss = self.calc_obj_loss(true_obj, pred_obj, ignore_mask)

        # YoloV1: Function (3)
        return xy_loss + wh_loss + class_loss + obj_loss, (xy_loss, wh_loss,
                                                           class_loss,
                                                           obj_loss)

    def calc_ignore_mask(self, true_obj, true_box, pred_box):
        # YOLOv3:
        # "If the bounding box prior is not the best but does overlap a ground
        # truth object by more than some threshold we ignore the prediction,
        # following [17]. We use the threshold of .5."
        # calculate the iou for each pair of pred bbox and true bbox, then find the best among them

        # (None, 13, 13, 3, 4)
        true_box_shape = tf.shape(true_box)
        # (None, 13, 13, 3, 4)
        pred_box_shape = tf.shape(pred_box)
        # (None, 507, 4)
        true_box = tf.reshape(true_box, [true_box_shape[0], -1, 4])
        # sort true_box to have non-zero boxes rank first
        true_box = tf.sort(true_box, axis=1, direction="DESCENDING")
        # (None, 100, 4)
        # only use maximum 100 boxes per groundtruth to calcualte IOU, otherwise
        # GPU emory comsumption would explode for a matrix like (16, 52*52*3, 52*52*3, 4)
        true_box = true_box[:, 0:100, :]
        # (None, 507, 4)
        pred_box = tf.reshape(pred_box, [pred_box_shape[0], -1, 4])

        # https://github.com/dmlc/gluon-cv/blob/06bb7ec2044cdf3f433721be9362ab84b02c5a90/gluoncv/model_zoo/yolo/yolo_target.py#L198
        # (None, 507, 507)
        iou = broadcast_iou(pred_box, true_box)
        # (None, 507)
        best_iou = tf.reduce_max(iou, axis=-1)
        # (None, 13, 13, 3)
        best_iou = tf.reshape(best_iou, [pred_box_shape[0], pred_box_shape[1], pred_box_shape[2], pred_box_shape[3]])
        # ignore_mask = 1 => don't ignore
        # ignore_mask = 0 => should ignore
        ignore_mask = tf.cast(best_iou < self.ignore_thresh, tf.float32)
        # (None, 13, 13, 3, 1)
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
        return ignore_mask

    def calc_obj_loss(self, true_obj, pred_obj, ignore_mask):
        """
        calculate loss of objectness: sum of L2 distances
        inputs:
        true_obj: objectness from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_obj: objectness from model prediction in shape of (batch, grid, grid, anchor, num_classes)
        outputs:
        obj_loss: objectness loss
        """
        obj_entropy = binary_cross_entropy(pred_obj, true_obj)

        obj_loss = true_obj * obj_entropy
        noobj_loss = (1 - true_obj) * obj_entropy * ignore_mask

        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3, 4))
        noobj_loss = tf.reduce_sum(
            noobj_loss, axis=(1, 2, 3, 4)) * self.lamda_noobj

        return obj_loss + noobj_loss

    def calc_class_loss(self, true_obj, true_class, pred_class):
        """
        calculate loss of class prediction
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_class: one-hot class from ground truth in shape of (batch, grid, grid, anchor, num_classes)
        pred_class: one-hot class from model prediction in shape of (batch, grid, grid, anchor, num_classes)
        outputs:
        class_loss: class loss
        """
        # Yolov1:
        # "Note that the loss function only penalizes classiﬁcation error
        # if an object is present in that grid cell (hence the conditional
        # class probability discussed earlier).
        class_loss = binary_cross_entropy(pred_class, true_class)
        class_loss = true_obj * class_loss
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3, 4))
        return class_loss

    def calc_xy_loss(self, true_obj, true_xy, pred_xy, weight):
        """
        calculate loss of the centroid coordinate: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_xy: centroid x and y from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_xy: centroid x and y from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        xy_loss: centroid loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)

        # in order to element-wise multiply the result from tf.reduce_sum
        # we need to squeeze one dimension for objectness here
        true_obj = tf.squeeze(true_obj, axis=-1)

        # YoloV1:
        # "It also only penalizes bounding box coordinate error if that
        # predictor is "responsible" for the ground truth box (i.e. has the
        # highest IOU of any predictor in that grid cell)."
        xy_loss = true_obj * xy_loss * weight

        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3)) * self.lambda_coord

        return xy_loss

    def calc_wh_loss(self, true_obj, true_wh, pred_wh, weight):
        """
        calculate loss of the width and height: sum of L2 distances
        inputs:
        true_obj: if the object present from ground truth in shape of (batch, grid, grid, anchor, 1)
        true_wh: width and height from ground truth in shape of (batch, grid, grid, anchor, 2)
        pred_wh: width and height from model prediction in shape of (batch, grid, grid, anchor, 2)
        weight: weight adjustment, reward smaller bounding box
        outputs:
        wh_loss: width and height loss
        """
        # shape (batch, grid, grid, anchor), eg. (32, 13, 13, 3)
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        true_obj = tf.squeeze(true_obj, axis=-1)
        wh_loss = true_obj * wh_loss * weight
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3)) * self.lambda_coord
        return wh_loss