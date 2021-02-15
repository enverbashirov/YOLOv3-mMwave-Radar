YOLO-mMwave-Radar

# README

TO BE WRITTEN

## TODO

15.02.2021 - EB

15.1.2021 - EB
- Get the mmwave data
- Finish training input/output pipeline
- Save the model and get the weights on mmwave data
- Get output frames

## ChangeLog

15.1.2021 - EB
- Working on ``train.py`` which is the training module of the network.\
[PyTorch Network Tranining Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- Added ``detect.py`` file which is used for input and output pipelining (taking input images, creating output images with bbs). Check ``arg_parse()`` function for input commands. Usage:
``python detect.py --images dog-cycle-car.png --det det``

13.1.2021 - EB
- Added ``Supporter`` class in "dataprep/utils.py". Bounding box calculation for ground truth data is ``label2bb()`` and a function for plotting with/without BB is ``plotRaw()``. Didn't compile the file, it should work though.