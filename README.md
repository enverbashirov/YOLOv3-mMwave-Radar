YOLO-mMwave-Radar

# README

TO BE WRITTEN

## ChangeLog

15.1.2021 - EB
- Added ``detect.py`` file which is used for input and output pipelining (taking input images, creating output images with bbs). Check ``arg_parse()`` function for input commands. Usage:
``python detect.py --images dog-cycle-car.png --det det``

13.1.2021 - EB
- Added ``Supporter`` class in "dataprep/utils.py". Bounding box calculation for ground truth data is ``label2bb()`` and a function for plotting with/without BB is ``plotRaw()``. Didn't compile the file, it should work though.