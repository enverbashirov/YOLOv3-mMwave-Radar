YOLO-mMwave-Radar

# README

Some Sources

[YOLOv3: An Incremental Improvement (paper)](https://arxiv.org/abs/1804.02767)

[YOLOv3 PyTorch](https://github.com/ecr23xx/yolov3.pytorch/blob/master/src/layers.py)

[YOLOv3 PyTorch (detection)](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

[PyTorch Network Tranining Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

[YOLOv3 Tensorflow](https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py)

[YOLOv3 Tensorflow (alternative)](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)



- FOLDER STRUCTURE
```
.
├── ...
├── cfg                    # DarkNet configuration files
├── checkpoints            # NEW!!! Training checkpoints
├── dataprep               # Data preprocessing files
├── raw                    # Raw dataset
├── save
│   ├── jp
│   │   ├── chext          # Images after `channel_extraction`
│   │   ├── final          # Network-ready images (`dataprep.plot4train()`)
│   │   ├── proc           # Images after `processing`
│   │   └── processed      # Images with point cloud and radar view (`dataprep.plot()`)
│   └── jp2
├── yolo                   # Network runner files
└── ...
```

- Network output params: (`Batch x No of BBs x BB attributes`)
    - `Batch Size`: number of images fed as a batch (e.g 8)
    - `No of BBs`: number of bounding boxes found for each image (e.g 10647 (usually))
    - `BB attributes`: (e.g 6) `bb_dims` (4) + `obj_score` (1) + `class_scores` (e.g 1 (number of objects)) 

## TODO

- [ ] Output pipeline
- [ ] Apply Non-Max Suppression (on model output)
    - to reduce output params before loss function to a single bb (as given in label)
- [ ] Detection (a working version)

- [ ] Hyperparameters check
- [ ] Objectiveness score loss calculation original uses binary cross entropy, we are using mean squared
- [ ] See if class score can be added
- [ ] Total loss may be wrong (some of the inputs in the batch are skipped due to empty labels while they may have been included during calculations of total loss (mean of the batch))
- [ ] Remove empty labelled data completely

## ChangeLog

21.03.2021 - EB
- Changed `lr` of `optim.SGD()` to 0.0001
- [x] Reduce the network
    - Reduced number of layers from 106 to 52 (best we can do without reducing the `YOLO` layers)
    - Computation time is reduced by ~1/3
- [x] Save the model and get weights for detection (
    - `yolo.util.save_checkpoint()`, `yolo.util.load_checkpoint()` (for training)
    - `yolo.darknet.load_weights()` (for detections, still to be tested))
- [x] Check if network output bounding box attributes are relative to the center of the prediction

18.03.2021 - EB
- Filtering the empty labels with `collate()` at `MmwaveDataset`
- Removed 'class' score attribute from everywhere

17.03.2021 - EB
- Added new `.\checkpoints` folder for saving network training status
- Loss is returning 'nan' after 2nd or 3rd iteration

16.03.2021 - EB
- Label bounding box is fine now
- Label class format should be fixed
- Non-training seems to be working
- [x] `YOLOLoss()`: Loss function
- [x] `NMSLayer()`: Non-max suppression

05.03.2021 - EB
- Working `torch.autograd` and `loss.backward()`

25.02.2021 - EB
- [x] Network training (a working version)
- [x] Input pipeline
- Didn't remove classes after all. Now there is only 1 class (person)
- Need some work on the network to raise the performance

22.02.2021 - EB
- Input doesn't match the parameter size for some reason
- Rest of the input pipelining is done!

16.02.2021 - EB
- `yolotrain.py` not quite working at the moment, almost there
- bb are a part of the filename now
- Dataset shuffling for train and test sets

15.02.2021 - EB
- Pre-proccessing should be done.
- Package `dataprep` is added. `dataprep.py` is the main data preparation file now.

15.01.2021 - EB
- Working on `train.py` which is the training module of the network.\
 Added `detect.py` file which is used for input and output pipelining (taking input images, creating output images with bbs). Check `arg_parse()` function for input commands. Usage:
`python detect.py --images dog-cycle-car.png --det det`

13.01.2021 - EB
- Added `Supporter` class in "dataprep/utils.py". Bounding box calculation for ground truth data is `label2bb()` and a function for plotting with/without BB is `plotRaw()`. Didn't compile the file, it should work though.
