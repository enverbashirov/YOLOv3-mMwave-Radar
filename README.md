YOLO-mMwave-Radar

# README

TO BE WRITTEN

## TODO

- Finish training input pipeline
- Save the model and get the weights on mmwave data
- Finish training output pipeline
- Get output frames

## ChangeLog

16.03.2021 - EB
- Package ``yolo`` is added. ``yolo.py`` is the main network training file now.
- Added ``cfg\yolotiny.cfg`` which is a smaller version of yolov3.

15.02.2021 - EB
- Pre-proccessing should be done.
- Package ``dataprep`` is added. ``dataprep.py`` is the main data preparation file now.

15.1.2021 - EB
- Working on ``train.py`` which is the training module of the network.\
[PyTorch Network Tranining Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- Added ``detect.py`` file which is used for input and output pipelining (taking input images, creating output images with bbs). Check ``arg_parse()`` function for input commands. Usage:
``python detect.py --images dog-cycle-car.png --det det``

13.1.2021 - EB
- Added ``Supporter`` class in "dataprep/utils.py". Bounding box calculation for ground truth data is ``label2bb()`` and a function for plotting with/without BB is ``plotRaw()``. Didn't compile the file, it should work though.