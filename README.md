YOLO-mMwave-Radar

# README

TO BE WRITTEN

## TODO

[x] Input pipeline
[x] Network training (a working version)
[ ] Save the model and get the weights on mmwave data
[ ] Output pipeline
[ ] Detection on an input

## ChangeLog

25.02.2021 - EB
- Training is done
- Didn't remove classes after all. Now there is only 1 class (person)
- Need some work on the network to raise the results

22.02.2021 - EB
- Input doesn't match the parameter size for some reason
- Rest of the input pipelining is done!

16.02.2021 - EB
- ``yolotrain.py`` not quite working at the moment, almost there
- bb are a part of the filename now
- Dataset shuffling for train and test sets

15.02.2021 - EB
- Pre-proccessing should be done.
- Package ``dataprep`` is added. ``dataprep.py`` is the main data preparation file now.

15.01.2021 - EB
- Working on ``train.py`` which is the training module of the network.\
[PyTorch Network Tranining Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

- Added ``detect.py`` file which is used for input and output pipelining (taking input images, creating output images with bbs). Check ``arg_parse()`` function for input commands. Usage:
``python detect.py --images dog-cycle-car.png --det det``

13.01.2021 - EB
- Added ``Supporter`` class in "dataprep/utils.py". Bounding box calculation for ground truth data is ``label2bb()`` and a function for plotting with/without BB is ``plotRaw()``. Didn't compile the file, it should work though.
