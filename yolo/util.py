from __future__ import division
import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# from torch.autograd import Variable

import os, random
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt

def draw_prediction(img_path, prediction, target, reso, names, save_path, video=False):
    """Draw prediction result

    Args
    - img_path: (str) Path to image
    - prediction: (np.array) Prediction result with size [#bbox, 8]
        8 = [batch_idx, x1, y1, x2, y2, objectness, cls_conf, class idx]
    - target: (np.array) Prediction result with size [#bbox, 5]
        8 = [batch_idx, x1, y1, x2, y2, class idx]
    - reso: (int) Image resolution
    - names: (list) Class names
    - save_path: (str) Path to save prediction result
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso
    draw = ImageDraw.Draw(img)

    # Drawing targets (labels)
    try:
        for i in range(target.shape[0]):
            bbox = target[i, 0:4].numpy()
            bbox = xywh2xyxy(bbox, target=True)
            caption = f'truth #{i}'

            color = (255, 255, 255)
            x1, y1, x2, y2 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
            draw.rectangle(((x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio)),
                           outline=color, width=2)
            draw.rectangle((x1 * w_ratio, y2 * h_ratio + 15,
                            x2 * w_ratio, y2 * h_ratio),
                           fill=color)
            draw.text((x1 * w_ratio + 2, y2 * h_ratio),
                      caption, fill='black')
    except Exception:
        print(f'[ERR] TEST | Could not draw target')

    # Drawing predictions
    try:
        for i in range(prediction.shape[0]):
            bbox = prediction[i, 1:5]
            conf = '%.2f' % prediction[i, -3]
            caption = f'pred {conf}'

            color = (0, 0, 255)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            draw.rectangle(((x1 * w_ratio, y1 * h_ratio, x2 * w_ratio, y2 * h_ratio)),
                           outline=color, width=int(1+prediction[i, -3]*5))
            draw.rectangle((x1 * w_ratio, y1 * h_ratio - 15,
                            x2 * w_ratio, y1 * h_ratio),
                           fill=color)
            draw.text((x1 * w_ratio + 2, y1 * h_ratio - 15),
                      caption, fill='white')
    except Exception:
        print(f'[ERR] TEST | Could not draw prediction')

    # img.show()
    img.save(save_path)
    img.close()

def animate_predictions(pathin, pathout, savetype='gif'):
    if savetype == 'gif':
        gif = []
        images = (Image.open(f'{pathin}/{f}').copy() for f in sorted(os.listdir(pathin)) if f.endswith('.png'))
        for image in images:
            gif.append(image)
        gif[0].save(f'{pathout}/sequence.gif', save_all=True, \
            optimize=False, append_images=gif[1:], loop=0, \
            duration=67)
    # elif savetype == 'mp4':
    #     videodims = (100,100)
    #     fourcc = cv2.VideoWriter_fourcc(*'avc1')    
    #     video = cv2.VideoWriter("test.mp4",fourcc, 60,videodims)
    #     img = Image.new('RGB', videodims, color = 'darkred')
    #     #draw stuff that goes on every frame here
    #     for i in range(0,60*60):
    #         imtemp = img.copy()
    #         # draw frame specific stuff here.
    #         video.write(cv2.cvtColor(np.array(imtemp), cv2.COLOR_RGB2BGR))
    #     video.release()

def IoU(box1, box2):
    """ Compute IoU between box1 and box2 """

    if box1.is_cuda == True:
        box1 = box1.cpu()
    if box2.is_cuda == True:
        box2 = box2.cpu()

    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou
    
def xywh2xyxy(bbox, target=False):
    if target:
        xc, yc = bbox[0], bbox[1]
        half_w, half_h = bbox[2] / 2, bbox[3] / 2
        return [xc - half_w, yc - half_h, xc + half_w, yc + half_h]
    
    bbox_ = bbox.clone()
    if len(bbox_.size()) == 1:
        bbox_ = bbox_.unsqueeze(0)
    xc, yc = bbox_[..., 0], bbox_[..., 1]
    half_w, half_h = bbox_[..., 2] / 2, bbox_[..., 3] / 2
    bbox_[..., 0] = xc - half_w
    bbox_[..., 1] = yc - half_h
    bbox_[..., 2] = xc + 2 * half_w
    bbox_[..., 3] = yc + 2 * half_h
    return bbox_

def xyxy2xywh(bbox, target=False):
    if target:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        xc, yc = bbox[0] + w/2, bbox[1] + h/2
        return [xc, yc, w, h]
    
    bbox_ = bbox.clone()
    if len(bbox_.size()) == 1:
        bbox_ = bbox_.unsqueeze(0)
    w, h = bbox_[..., 2] - bbox_[..., 0], bbox_[..., 3] - bbox_[..., 1]
    xc, yc = bbox_[..., 0] + w/2, bbox_[..., 1] + h/2

    bbox_[..., 0] = xc
    bbox_[..., 1] = yc
    bbox_[..., 2] = w
    bbox_[..., 3] = h
    return bbox_

def load_checkpoint(checkpoint_dir, epoch, iteration):
    """Load checkpoint from path

    Args
    - checkpoint_dir: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint
    - iteration: (int) iteration of checkpoint in one epoch

    Returns
    - start_epoch: (int)
    - start_iteration: (int)
    - state_dict: (dict) state of model
    """
    path = os.path.join(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    if not os.path.isfile(path):
        raise Exception("Checkpoint in epoch %d doesn't exist" % epoch)

    checkpoint = torch.load(path)
    start_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    start_iteration = checkpoint['iteration']

    assert epoch == start_epoch, "epoch != checkpoint's start_epoch"
    assert iteration == start_iteration, "iteration != checkpoint's start_iteration"
    return start_epoch, start_iteration, state_dict

def save_checkpoint(checkpoint_dir, epoch, iteration, save_dict):
    """Save checkpoint to path

    Args
    - path: (str) absolute path to checkpoint folder
    - epoch: (int) epoch of checkpoint file
    - iteration: (int) iteration of checkpoint in one epoch
    - save_dict: (dict) saving parameters dict
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, str(epoch) + '.' + str(iteration) + '.ckpt')
    assert epoch == save_dict['epoch'], "[ERROR] epoch != save_dict's start_epoch"
    assert iteration == save_dict['iteration'], "[ERROR] iteration != save_dict's start_iteration"
    if os.path.isfile(path):
        print("[WARNING] Overwrite checkpoint in epoch %d, iteration %d" %
              (epoch, iteration))
    try:
        torch.save(save_dict, path)
    except Exception:
        raise Exception("[ERROR] Fail to save checkpoint")

    print("[LOG] Checkpoint %d.%d.ckpt saved" % (epoch, iteration))

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                        # store the lines in a list
    lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']              # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]           # get rid of fringe whitespaces
    file.close()
    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               # This marks the start of a new block
            if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                blocks.append(block)     # add it the blocks list
                block = {}               # re-init the block
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks

def plot_losses(tlosses, vlosses=None, save=True):

    plt.plot(range(0, len(tlosses)), tlosses)
    if vlosses:
        plt.plot(range(0, len(vlosses)), vlosses)
        plt.legend(['Train loss', 'Valid loss'], loc='upper left')
        plt.title(f'Training and Validation loss ({len(tlosses)} Epochs) ')
    else:
        plt.legend(['Train loss'], loc='upper left')
        plt.title(f'Training loss ({len(tlosses)} Epochs) ')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if save:
        plt.savefig(f'save/results/loss_{len(tlosses)}.png', dpi=100)
        print(f'[LOG] TRAIN | Loss graph save \"save/results/loss_{len(tlosses)}.png\"')
    else:
        plt.show()
    plt.close()