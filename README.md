YOLO-mMwave-Radar

# README

#### Usage

Help: `python . --help`\
Data preprocessing: `python . dataprep --help`\
e.g `python . dataprep --pathin <in_dir> --pathout <out_dir> --chext --proc --truth`\
Training: `python . train --help`\
e.g `python . train --pathin trainset --datasplit 0.9 --ckpt 80.0 --ep 500`\
Prediction: `python . predict --help`\
e.g `python . predict --pathin testset3 --pathout test --ckpt 80.0 --nms 0.001 --obj 0.005 --video gif`\

#### Some Sources

[YOLOv3: An Incremental Improvement (paper)](https://arxiv.org/abs/1804.02767) \
[YOLOv3 PyTorch](https://github.com/ecr23xx/yolov3.pytorch/blob/master/src/layers.py) \
[YOLOv3 PyTorch (detection)](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/) \
[PyTorch Network Tranining Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) \
[YOLOv3 Tensorflow](https://github.com/qqwweee/keras-yolo3/blob/master/yolo3/model.py) \
[YOLOv3 Tensorflow (alternative)](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e)\

#### FOLDER STRUCTURE
```
.
├── ...
├── cfg                    # DarkNet config files
├── dataprep               # LIBRARY: Preprocessing
├── dataset                # DATASETS
│   ├── <set_name>         `dataprep --pathin <set_name>
│   │   ├── chext          # Images after channel extraction `dataprep --chext`
│   │   ├── final          # Network-ready images `dataprep --truth`
│   │   └── proc           # Images after `dataprep --proc`
├── raw                    # Raw dataset files (.h5)
├── results                # PREDICTIONS 
│   ├── <set_name>         `predict --pathin <set_name>`
│   │   └── pred           # Images with predicted and true bbs
├── save
│   └── checkpoints        # Model checkpoints
├── yolo                   # LIBRARY: Object detection (train and predict)
└── ...
```

#### Documentation

- Network output params: (`Batch x No of BBs x BB attributes`)
    - `Batch Size`: number of images fed as a batch (e.g 8)
    - `No of BBs`: number of bounding boxes found for each image with full network config (e.g 10647 (usually))
    - `BB attributes`: (e.g 6) `bb_dims` (4) + `obj_score` (1) + `class_scores` (e.g 1 (number of objects)) 


## TODO & NOTES

- Util (`yolo.util`)
	- mAP (`mean_average_precision()`) over range of IoUs
    - [mAP with NMS](https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522) \
    - [mAP](https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b) \
	- mAP over epoch plot (`plot_mAP()`)
- Hyperparameters check
    - [RayTune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)

#### Changes Required (temporal info)

- Total number of parameters doesn't include GRU section

- mAP calculation only takes IoU into account
    - [ ] Implement objectiveness score precision/recall calculation
	

## ChangeLog

12.08.2021 - EB - Version 1.3.5
- Optimizations for GRU network
    - Autoencoder/decoder logic for dimensionality reduction
    - Merging some layers after GRU with previous

10.08.2021 - EB - Version 1.3.4
- Implemented RNN (GRU) logic
    - [x] Add GRU state layer to use temporal information
- Implemented image augmentation

04.08.2021 - EB - Version 1.3.3
- Working on RNN implementation
- Working on image augmentations
- Implemented sequence training
- Added `scripts/recalculatebbox.py` which recalculates the label bounding boxes in a better way.

23.06.2021 - EB - Version 1.3.2
- Implemented `plot_precision_recall()`

25.05.2021 - EB - Version 1.3.1
- Working on `plot_precision_recall()`
- Implemented `correctness()` for TP/FP/FN/TN calculations
- Implemented `precision_recall()` for cumulative TP and FP, precision and recall calculations

08.04.2021 - EB - Version 1.2
- Images to avi
- Fixed multi bb ground truth
- Fixed folder structure to final version

07.04.2021 - EB - Version 1.1
- Images to gif
    - [x] Animating results
- [x] Small truth bb issue may be existing (on w, h translation (matplotlib to PIL?))

05.04.2021 - EB - Finalized dataprep
- Fixed shuffling in `yolo.dataset`
- Default learning rate is reduced to 1e-5 from 1e-4
- `dataprep` is stable
    - `python . dataprep --help`

31.03.2021 - EB - Version 1.0
- Added `__main__`
    - Check `python . --help`
    - Example train run: `python . train --lr 0.00001 --ep10`
    - Example predict run: `python . predict --cfg test --pathout test/results --ckpt 3.0 --obj 0.2 --nms 0.5`
    - Example dataprep run: `python . data`
- Renamed `custom.cfg` as `yolov3micro.cfg`
- Removed class score (`cls`) from loss calculation as we have only 1 class
- Changed objectiveness (`obj`) loss calculation from MSELoss to BCELoss
    - [x] ~~Objectiveness score loss calculation original uses binary cross entropy, we are using mean squared~~
- Fixed bb calculation/scale issue
    - [x] ~~Total loss may be wrong (some inputs were skipping due to empty labels)~~
- [x] On validation loss, keep history and add graphs
    - `yolo.util.plot_losses()`
- Added some random image manipulations/transformations for training input
    - [x] Check the torchvision.transforms functionality
- [x] Remove empty labelled data completely
- Moved and renamed `dataprep.py` to `./dataprep` as `truth.py`
- Fixed functionality of batch prediction

25.03.2021 - EB - First version
- Reintroducing class and class loss
- `yolo.getDataLoaders()`: dataset allocation for train/val or single set
    - with `random_seed` parameter we can get the same shuffle everytime (useful for testing)
- Validation is now carried out right after each epoch
- [x] Output pipeline
- [x] Apply Non-Max Suppression
- [x] Detection (a working version)

23.03.2021 - EB - custom network
- Changed `lr` of `optim.SGD()` to 0.0001
- [x] Reduce the network
    - Reduced number of layers from 106 to 52 (best we can do without reducing the `YOLO` layers)
    - Computation time is reduced by ~1/3
- [x] Save the model and get weights for detection
    - `yolo.util.save_checkpoint()`, `yolo.util.load_checkpoint()` (for training)
    - `yolo.darknet.load_weights()` (for detections, still to be tested)
- [x] Check if network output bounding box attributes are relative to the center of the prediction

18.03.2021 - EB - Learning
- Filtering the empty labels with `collate()` at `MmwaveDataset`
- Removed 'class' score attribute from everywhere

17.03.2021 - EB - Filtering empty labels
- Added new `./checkpoints` folder for saving network training status
- Loss is returning 'nan' after 2nd or 3rd iteration

16.03.2021 - EB - Training
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


07.04.2021 - EB - Version 1.1
- Predictions to gif
    - [x] Animating results
- [x] Small truth bb issue may be existing (on w, h translation (matplotlib to PIL?))

05.04.2021 - EB - Finalized dataprep
- Fixed shuffling in `yolo.dataset`
- Default learning rate is reduced to 1e-5 from 1e-4
- `dataprep` is stable
    - `python . dataprep --help`

31.03.2021 - EB - Version 1.0
- Added `__main__`
    - Check `python . --help`
    - Example train run: `python . train --lr 0.00001 --ep10`
    - Example predict run: `python . predict --cfg test --pathout test/results --ckpt 3.0 --obj 0.2 --nms 0.5`
    - Example dataprep run: `python . data`
- Renamed `custom.cfg` as `yolov3micro.cfg`
- Removed class score (`cls`) from loss calculation as we have only 1 class
- Changed objectiveness (`obj`) loss calculation from MSELoss to BCELoss
    - [x] ~~Objectiveness score loss calculation original uses binary cross entropy, we are using mean squared~~
- Fixed bb calculation/scale issue
    - [x] ~~Total loss may be wrong (some inputs were skipping due to empty labels)~~
- [x] On validation loss, keep history and add graphs
    - `yolo.util.plot_losses()`
- Added some random image manipulations/transformations for training input
    - [x] Check the torchvision.transforms functionality
- [x] Remove empty labelled data completely
- Moved and renamed `dataprep.py` to `./dataprep` as `truth.py`
- Fixed functionality of batch prediction

24.03.2021 - EB
- Reintroducing class and class loss
- `yolo.getDataLoaders()`: dataset allocation for train/val or single set
    - with `random_seed` parameter we can get the same shuffle everytime (useful for testing)
- Validation is now carried out right after each epoch
- [x] Output pipeline
- [x] Apply Non-Max Suppression
- [x] Detection (a working version)

21.03.2021 - EB
- Changed `lr` of `optim.SGD()` to 0.0001
- [x] Reduce the network
    - Reduced number of layers from 106 to 52 (best we can do without reducing the `YOLO` layers)
    - Computation time is reduced by ~1/3
- [x] Save the model and get weights for detection
    - `yolo.util.save_checkpoint()`, `yolo.util.load_checkpoint()` (for training)
    - `yolo.darknet.load_weights()` (for detections, still to be tested)
- [x] Check if network output bounding box attributes are relative to the center of the prediction

18.03.2021 - EB
- Filtering the empty labels with `collate()` at `MmwaveDataset`
- Removed 'class' score attribute from everywhere

17.03.2021 - EB
- Added new `./checkpoints` folder for saving network training status
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
