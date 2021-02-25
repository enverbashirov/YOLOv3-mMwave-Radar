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