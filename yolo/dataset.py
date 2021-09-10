import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
# from torchvision import transforms
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from .util import *

# anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
#                        [59, 119], [116, 90], [156, 198], [373, 326]],
#                       np.float32) / 416

class MmwaveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size = 0, reso=416, transforms = None, seq = 1):
        files = sorted(os.listdir(data_dir))
        self.files = [f"{data_dir}/{x}" for x in files]
        # self.files = self.files[:int(len(self.files)/100)]

        if data_size < 0 or data_size > len(self.files):
            assert("Data size should be between 0 to number of files in the dataset")
        
        if data_size == 0:
            data_size = len(self.files)-seq
        
        self.data_size = data_size
        self.reso = reso
        self.transforms = transforms
        self.seq = seq
  
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_paths, images, images_, labels, labels_, bbs = [], [], [], [], [], []
        for s in range(0, self.seq):
            image_path = self.files[idx+s]
            image = Image.open(image_path)
            img_w, img_h = image.size
            labels_str = image_path.split("_")[-1]
            
            image_paths.append(image_path)
            label = xywh2xyxy(self.getLabel(labels_str, img_w, img_h), target=True)
            image = np.reshape(np.array(image.convert('RGB')),(self.reso,self.reso,3))
            images.append(image.astype(np.float32))
            labels_.append(label)
        labels = np.array(labels_)[:,:,:4]

        bbs = []
        for l in labels:
            bbs.append(BoundingBoxesOnImage.from_xyxy_array(l, shape=(self.reso, self.reso)))

        images, bbs = self.preProcessImage(images, bbs)

        for i in range(0, len(images)):
            # ia.imshow(bbs[i].draw_on_image(Image.fromarray(np.uint8(images[i])).convert('RGB'), size=2, color=0))
            label = xyxy2xywh(bbs[i].to_xyxy_array(), target=True)
            label = self.normalizeLabels(label, img_w, img_h)
            temp = []
            for j in range(0, len(label)):
                temp.append(np.append(label[j], labels_[i][j][4:]))
            labels_[i] = torch.tensor(temp)
            images_.append(torch.tensor(np.reshape(images[i], (3,self.reso,self.reso))))
        # exit()

        return image_paths, torch.stack(images_), torch.stack(labels_)

    # Get label bounding boxes
    def getLabel(self, labels_str, img_w, img_h):
        if "[[" in labels_str:
            labels_str = labels_str.split('[[')[1].split(']]')[0].split('],[')
            labels = np.zeros((4, 5))
            for i, l in enumerate(labels_str):
                # Get labels
                label = np.zeros(5)
                label[:4] = np.array([int(a) for a in l.split(',')]) # [xc, yc, w, h]
                labels[i, :] = label

        else:
            labels_str = labels_str.split('[')[1].split(']')[0].split(',') # get the bb info from the filename
            
            # Get labels
            labels = np.zeros((1, 5))
            labels[0, :4] = np.array([int(a) for a in labels_str]) # [xc, yc, w, h]
            
            if np.any(labels[0, :4] == 0):
                return None

        return labels

    # Normalize labels
    def normalizeLabels(self, labels, img_w, img_h):
        if len(np.array(labels).shape) == 2:
            for label in labels:
                label[0] /= img_w #Xcenter
                label[1] /= img_h #Ycenter
                label[2] /= img_w #Width
                label[3] /= img_h #Height
        else:
            labels[0] /= img_w #Xcenter
            labels[1] /= img_h #Ycenter
            labels[2] /= img_w #Width
            labels[3] /= img_h #Height
        
        return labels

    #Image custom preprocessing if required
    def preProcessImage(self, images, labels=None):
        if self.transforms:
            seq = iaa.Sequential([
                iaa.Fliplr(0.1), # horizontal flips
                iaa.Crop(px=(0, 32)), # random crops

                iaa.Sometimes(0.05, iaa.SaltAndPepper(0.05)),
                iaa.Dropout(p=(0, 0.05)), # dropout some pixels

                iaa.Sometimes(0.01, iaa.GaussianBlur(sigma=(0, 0.5))),

                # Strengthen or weaken the contrast in each image.
                # iaa.Sometimes(0.2, iaa.LinearContrast((0.9, 1.01))),

                # Add gaussian noise.
                iaa.Sometimes(0.02,
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.2),
                ),

                # Make some images brighter and some darker.
                iaa.Sometimes(0.1, iaa.Multiply((0.9, 1.1), per_channel=0.1)),

                # Scale/zoom
                iaa.Sometimes(0.1, iaa.Affine(scale={"x": (0.9, 1.0), "y": (0.9, 1)})),
                # Translate/move
                iaa.Sometimes(0.1, iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)})),
                # Rotate
                iaa.Sometimes(0.1, iaa.Affine(rotate=(-5, 5))),
                # Shear
                iaa.Sometimes(0.1, iaa.Affine(shear=(-3, 3)))
            ], random_order=True) # apply augmenters in random order

            seq = seq.to_deterministic()
            images_, labels_ = [], []
            for i, l in zip(images, labels):
                images_.append(seq.augment_image(i))
                labels_.append(seq.augment_bounding_boxes(l))
                
            return images_, labels_
        else:
            seq = iaa.Identity()
            
            return seq(images=images, bounding_boxes=labels)

def collate(batch):
    batch = list(filter(lambda x:x[1] is not None, batch))
    return default_collate(batch) # Use the default method to splice the filtered batch data

def getDataLoaders(data_dir, transforms, reso=416, train_split=0, batch_size=8, seq=1, \
    num_workers=2, collate_fn=collate, random_seed=0):
    
    if train_split < 0 or train_split > 1:
        raise Exception(f"data_loader | Split ({train_split}) coefficient should be 0 < x < 1")

    dataset = MmwaveDataset(data_dir=data_dir, reso=reso, transforms=transforms, seq=seq)
    shuffle = True if random_seed != 0 else False
    
    # Single Set
    if train_split == 0 or train_split == 1:
        return None, torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers, collate_fn = collate_fn)

    # Generate a fixed seed
    generator = torch.Generator()
    if random_seed != 0:
        generator.manual_seed(random_seed)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    
    trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

    # Train and Validation sets
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, collate_fn = collate_fn),   \
            torch.utils.data.DataLoader(testset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=num_workers, collate_fn = collate_fn)
