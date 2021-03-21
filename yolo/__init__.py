import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

import os, pickle, random, time
import numpy as np
from PIL import Image

from .darknet import *
from .util import *

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
        img_w, img_h = image.size
        
        image = self.preProcessImage(image)
        image = image.astype(np.float32)
        if self.transforms:
            image = self.transforms(image)

        labels = np.zeros((1, 5)) # to make it array of bbs (for multiple bbs in the future)
        labels_str = image_address.split("_") \
            [-1].split('[')[1].split(']')[0].split(',') # get the bb info from the filename
        labels[0, :4] = np.array([int(a) for a in labels_str]) # [xc, yc, w, h]
        
        if np.any(labels[0, :4] == 0):
            return image, None

        # Normalizing labels
        labels[0, 0] /= img_w #Xcenter
        labels[0, 1] /= img_h #Ycenter
        labels[0, 2] /= img_w #Width
        labels[0, 3] /= img_h #Height
        # print(labels_str, labels)
        # labels[0, 4] = 0 # class label (0 = person)

        return image, labels

    #Image custom preprocessing if required
    def preProcessImage(self, image):

        image = np.array(image.convert('RGB'))
        return image.transpose(2,1,0)

def collate(batch):
    batch = list(filter(lambda x:x[1] is not None, batch))
    return default_collate(batch) # Use the default method to splice the filtered batch data
