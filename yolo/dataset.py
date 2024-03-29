import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
# from torchvision import transforms

import os
# import random
import numpy as np
from PIL import Image

# anchors_wh = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
#                        [59, 119], [116, 90], [156, 198], [373, 326]],
#                       np.float32) / 416

class MmwaveDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size = 0, transforms = None):
        files = sorted(os.listdir(data_dir))
        self.files = [f"{data_dir}/{x}" for x in files]
        
        if data_size < 0 or data_size > len(files):
            assert("Data size should be between 0 to number of files in the dataset")
        
        if data_size == 0:
            data_size = len(files)
        
        self.data_size = data_size
        self.transforms = transforms
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path)
        img_w, img_h = image.size
        
        image = self.preProcessImage(image)

        labels = [] # to make it array of bbs (for multiple bbs in the future)
        labels_str = image_path.split("_")[-1]

        if "[[" in labels_str:
            labels_str = labels_str.split('[[')[1].split(']]')[0].split('],[')
            labels = np.zeros((4, 5))
            for i, l in enumerate(labels_str):
                label = np.zeros(5)
                label[:4] = np.array([int(a) for a in l.split(',')]) # [xc, yc, w, h]

                # Normalizing labels
                label[0] /= img_w #Xcenter
                label[1] /= img_h #Ycenter
                label[2] /= img_w #Width
                label[3] /= img_h #Height

                labels[i, :] = label
        else:
            labels_str = labels_str.split('[')[1].split(']')[0].split(',') # get the bb info from the filename
            labels = np.zeros((1, 5))
            labels[0, :4] = np.array([int(a) for a in labels_str]) # [xc, yc, w, h]
            
            if np.any(labels[0, :4] == 0):
                return image, None

            # Normalizing labels
            labels[0, 0] /= img_w #Xcenter
            labels[0, 1] /= img_h #Ycenter
            labels[0, 2] /= img_w #Width
            labels[0, 3] /= img_h #Height
            # labels[0, 4] = 0 # class label (0 = person)
            # print(torch.any(torch.isfinite(image) == False), labels)

        return image_path, image, labels

    #Image custom preprocessing if required
    def preProcessImage(self, image):
        image = image.convert('RGB')
        if self.transforms:
            return self.transforms(image)
        else:
            image = np.array(image)
            image = image.transpose(2,1,0)
            return image.astype(np.float32)

def collate(batch):
    batch = list(filter(lambda x:x[1] is not None, batch))
    return default_collate(batch) # Use the default method to splice the filtered batch data

def getDataLoaders(data_dir, transforms, train_split=0, batch_size=8, \
    num_workers=2, collate_fn=collate, random_seed=0):
    
    if train_split < 0 or train_split > 1:
        raise Exception(f"data_loader | Split ({train_split}) coefficient should be 0 < x < 1")

    dataset = MmwaveDataset(data_dir=data_dir, transforms=transforms)
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
        shuffle=shuffle, num_workers=2, collate_fn = collate_fn),   \
            torch.utils.data.DataLoader(testset, batch_size=batch_size, \
        shuffle=shuffle, num_workers=2, collate_fn = collate_fn)
