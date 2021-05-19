import torch
import numpy as np
import os

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

b1 = np.array([[1,1,1,1], [2,2,2,2], [3,3,3,3]])
print(b1[0,...,0])

a = 1
a += 1
print(a)

# b2 = np.array([1,1,1,1])
# box1 = torch.tensor(b1)
# box2 = torch.tensor(b2)

# print(box1, box2)

# print(IoU(box1, box2))

