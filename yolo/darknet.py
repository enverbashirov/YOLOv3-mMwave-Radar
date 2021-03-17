from __future__ import division

import torch, torch.nn as nn, torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
from pprint import pprint

from .util import *

# =================================================================
# MAXPOOL (with stride = 1, NOT SURE IF NEEDED)
class MaxPool1s(nn.Module):

    def __init__(self, kernel_size):
        super(MaxPool1s, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        return pooled_x

# EMPTY LAYER
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# YOLO / DETECTION LAYER
class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, reso, ignore_thresh):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.reso = reso
        self.ignore_thresh = ignore_thresh

    def forward(self, x, y_true=None):
        bs, _, gs, _ = x.size()
        stride = self.reso // gs  # no pooling used, stride is the only downsample
        num_attrs = 5 + self.num_classes  # tx, ty, tw, th, p0
        nA = len(self.anchors)
        scaled_anchors = torch.Tensor(
            [(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]).cuda()

        # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
        x = x.view(bs, nA, num_attrs, gs, gs).permute(
            0, 1, 3, 4, 2).contiguous()

        pred = torch.Tensor(bs, nA, gs, gs, num_attrs).cuda()

        pred_tx = torch.sigmoid(x[..., 0]).cuda()
        pred_ty = torch.sigmoid(x[..., 1]).cuda()
        pred_tw = x[..., 2].cuda()
        pred_th = x[..., 3].cuda()
        pred_conf = torch.sigmoid(x[..., 4]).cuda()
        if self.training == True:
            pred_cls = x[..., 5:].cuda()  # softmax in cross entropy
        else:
            pred_cls = F.softmax(x[..., 5:], dim=-1).cuda()  # class

        grid_x = torch.arange(gs).repeat(gs, 1).view(
            [1, 1, gs, gs]).float().cuda()
        grid_y = torch.arange(gs).repeat(gs, 1).t().view(
            [1, 1, gs, gs]).float().cuda()
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        pred[..., 0] = pred_tx + grid_x
        pred[..., 1] = pred_ty + grid_y
        pred[..., 2] = torch.exp(pred_tw) * anchor_w
        pred[..., 3] = torch.exp(pred_th) * anchor_h
        pred[..., 4] = pred_conf
        pred[..., 5:] = pred_cls

        if not self.training:
            pred[..., :4] *= stride
            return pred.view(bs, -1, num_attrs)
        else:
            loss = YOLOLoss([bs, nA, gs], scaled_anchors, self.num_classes, pred, [pred_tx, pred_ty, pred_tw, pred_th])
            loss = loss(x, y_true)
            return loss

# YOLOv3 Loss
class YOLOLoss(nn.Module):
    def __init__(self, shape, scaled_anchors, num_classes, pred, pred_t):
        super(YOLOLoss, self).__init__()
        self.bs = shape[0]
        self.nA = shape[1]
        self.gs = shape[2]
        self.scaled_anchors = scaled_anchors
        self.num_classes = num_classes
        self.predictions = pred
        self.pred_conf = pred[..., 4]
        self.pred_cls = pred[..., 5:]
        self.pred_tx = pred_t[0]
        self.pred_ty = pred_t[1]
        self.pred_tw = pred_t[2]
        self.pred_th = pred_t[3]

    def forward(self, x, y_true):
        gt_tx = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        gt_ty = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        gt_tw = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        gt_th = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        gt_conf = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        gt_cls = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()

        obj_mask = torch.zeros(self.bs, self.nA, self.gs, self.gs, requires_grad=False).cuda()
        for idx in range(self.bs):
            for y_true_one in y_true[idx]:
                y_true_one = y_true_one.cuda()
                gt_bbox = y_true_one[:4] * self.gs
                gt_cls_label = int(y_true_one[4])

                gt_xc, gt_yc, gt_w, gt_h = gt_bbox[0:4]
                gt_i = gt_xc.long().cuda()
                gt_j = gt_yc.long().cuda()

                pred_bbox = self.predictions[idx, :, gt_j, gt_i, :4]
                ious = IoU(xywh2xyxy(pred_bbox), xywh2xyxy(gt_bbox))
                best_iou, best_a = torch.max(ious, 0)

                w, h = self.scaled_anchors[best_a]
                gt_tw[idx, best_a, gt_j, gt_i] = torch.log(gt_w / w)
                gt_th[idx, best_a, gt_j, gt_i] = torch.log(gt_h / h)
                gt_tx[idx, best_a, gt_j, gt_i] = gt_xc - gt_i.float()
                gt_ty[idx, best_a, gt_j, gt_i] = gt_yc - gt_j.float()
                gt_conf[idx, best_a, gt_j, gt_i] = best_iou
                gt_cls[idx, best_a, gt_j, gt_i] = gt_cls_label

                obj_mask[idx, best_a, gt_j, gt_i] = 1

        MSELoss = nn.MSELoss(reduction='sum')
        # BCELoss = nn.BCELoss(reduction='sum')
        CELoss = nn.CrossEntropyLoss(reduction='sum')

        loss = dict()
        loss['x'] = MSELoss(self.pred_tx * obj_mask, gt_tx * obj_mask)
        loss['y'] = MSELoss(self.pred_ty * obj_mask, gt_ty * obj_mask)
        loss['w'] = MSELoss(self.pred_tw * obj_mask, gt_tw * obj_mask)
        loss['h'] = MSELoss(self.pred_th * obj_mask, gt_th * obj_mask)
        # loss['cls'] = BCELoss(pred_cls * obj_mask, cls_mask * obj_mask)

        loss['cls'] = CELoss((self.pred_cls * obj_mask.unsqueeze(-1)).view(-1, self.num_classes),
                                (gt_cls * obj_mask).view(-1).long())
        loss['conf'] = MSELoss(self.pred_conf * obj_mask * 5, gt_conf * obj_mask * 5) + \
            MSELoss(self.pred_conf * (1 - obj_mask), self.pred_conf * (1 - obj_mask))

        # pprint(loss)

        return loss

# Non-Max Suppression
class NMSLayer(nn.Module):
    """
    NMS layer which performs Non-maximum Suppression
    1. Filter background
    2. Get detection with particular class
    3. Sort by confidence
    4. Suppress non-max detection
    """

    def __init__(self, conf_thresh=0.5, nms_thresh=0.5):
        """
        Args:
        - conf_thresh: (float) fore-ground confidence threshold
        - nms_thresh: (float) nms threshold
        """
        super(NMSLayer, self).__init__()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def forward(self, x):
        """
        Args
          x: (Tensor) detection feature map, with size [bs, num_bboxes, 5 + nC]

        Returns
          predictions: (Tensor) detection result with size [num_bboxes, [image_batch_idx, 4 offsets, p_obj, max_conf, cls_idx]]
        """
        bs, _, _ = x.size()
        predictions = torch.Tensor().cuda()

        for idx in range(bs):
            pred = x[idx]

            try:
                non_zero_pred = pred[pred[:, 4] > self.conf_thresh]
                non_zero_pred[:, :4] = xywh2xyxy(non_zero_pred[:, :4])
                max_score, max_idx = torch.max(non_zero_pred[:, 5:], 1)
                max_idx = max_idx.float().unsqueeze(1)
                max_score = max_score.float().unsqueeze(1)
                non_zero_pred = torch.cat(
                    (non_zero_pred[:, :5], max_score, max_idx), 1)
                classes = torch.unique(non_zero_pred[:, -1])
            except Exception:  # no object detected
                print('No object detected')
                continue

            for cls in classes:
                cls_pred = non_zero_pred[non_zero_pred[:, -1] == cls]
                conf_sort_idx = torch.sort(cls_pred[:, 5], descending=True)[1]
                cls_pred = cls_pred[conf_sort_idx]
                max_preds = []
                while cls_pred.size(0) > 0:
                    max_preds.append(cls_pred[0].unsqueeze(0))
                    ious = IoU(max_preds[-1], cls_pred)
                    cls_pred = cls_pred[ious < self.nms_thresh]

                if len(max_preds) > 0:
                    max_preds = torch.cat(max_preds).data
                    batch_idx = max_preds.new(max_preds.size(0), 1).fill_(idx)
                    seq = (batch_idx, max_preds)
                    predictions = torch.cat(seq, 1) if predictions.size(
                        0) == 0 else torch.cat((predictions, torch.cat(seq, 1)))

        return predictions
# =================================================================

# NETWORK
class DarkNet(nn.Module):
    def __init__(self, cfgfile, reso):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.reso = reso
        self.net_info, self.module_list = self.create_modules(self.blocks)
        self.nms = NMSLayer()

    def forward(self, x, y_true=None, CUDA=False):
        modules = self.blocks[1:]
        predictions = torch.Tensor().cuda() if CUDA else torch.Tensor()
        outputs = dict()   #We cache the outputs for the route layer
        loss = dict()

        for i, module in enumerate(modules):
            if module["type"] == "convolutional" or module["type"] == "upsample":
                x = self.module_list[i](x)
                outputs[i] = x

            elif  module["type"] == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
                outputs[i] = x

            elif module["type"] == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
    
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
    
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
    
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
    
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
                
            elif module["type"] == 'yolo':
                if self.training == True:
                    loss_part = self.module_list[i][0](x, y_true)
                    for key, value in loss_part.items():
                        value = value
                        loss[key] = loss[key] + \
                            value if key in loss.keys() else value
                        loss['total'] = loss['total'] + \
                            value if 'total' in loss.keys() else value
                else:
                    x = self.module_list[i][0](x)
                    predictions = x if len(predictions.size()) == 1 else torch.cat(
                        (predictions, x), 1)
                outputs[i] = outputs[i-1]  # skip
            
            # Print the layer information
            # print(i, module["type"], x.shape)
        
        # return detection result only when evaluated
        if self.training == True:
            return loss
        else:
            predictions = self.nms(predictions)
            return predictions

    def create_modules(self, blocks):
        net_info = blocks[0]   #Captures the information about the input and pre-processing  
        # print(net_info)
        module_list = nn.ModuleList()
        in_channels = 3
        out_channels_list = []
        
        for index, block in enumerate(blocks[1:]):
            module = nn.Sequential()
        
            # Convolutional Layer
            if (block["type"] == "convolutional"):
                activation = block["activation"]
                try:
                    batch_normalize = int(block["batch_normalize"])
                    bias = False
                except:
                    batch_normalize = 0
                    bias = True
            
                out_channels = int(block["filters"])
                kernel_size = int(block["size"])
                padding = (kernel_size - 1) // 2 if int(block["pad"]) else 0
                stride = int(block["stride"])
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = bias)
                module.add_module("conv_{0}".format(index), conv)
            
                if batch_normalize:
                    bn = nn.BatchNorm2d(out_channels)
                    module.add_module("batch_norm_{0}".format(index), bn)

                if activation == "leaky":
                    activn = nn.LeakyReLU(0.1, inplace = True)
                    module.add_module("leaky_{0}".format(index), activn)
            
            # Up Sample Layer
            elif (block["type"] == "upsample"):
                stride = int(block["stride"]) # = 2 in Yolov3
                upsample = nn.Upsample(scale_factor = stride, mode = "nearest")
                module.add_module("upsample_{}".format(index), upsample)
                    
            # Shortcut Layer
            elif block["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module("shortcut_{}".format(index), shortcut)

            # Route Layer
            elif (block["type"] == "route"):
                route = EmptyLayer()
                module.add_module("route_{0}".format(index), route)

                block["layers"] = block["layers"].split(',')
                start = int(block["layers"][0])
                if len(block['layers']) == 1:
                    start = int(block['layers'][0])
                    out_channels = out_channels_list[index + start]
                elif len(block['layers']) == 2:
                    start = int(block['layers'][0])
                    end = int(block['layers'][1])
                    out_channels = out_channels_list[index + start] + out_channels_list[end]        
                
            # Yolo Layer
            elif block["type"] == "yolo":
                mask = block["mask"].split(",")
                mask = [int(x) for x in mask]
        
                anchors = block["anchors"].split(",")
                anchors = [int(a) for a in anchors]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                anchors = [anchors[i] for i in mask]
        
                num_classes = int(block['classes'])
                ignore_thresh = float(block['ignore_thresh'])

                detection = YOLOLayer(anchors, num_classes, self.reso, ignore_thresh)
                module.add_module("Detection_{}".format(index), detection)
                                
            module_list.append(module)
            in_channels = out_channels
            out_channels_list.append(out_channels)
            
        return (net_info, module_list)