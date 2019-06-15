# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation. This file is a reversion of yolov3 and most of the py files are coming from another github author. I test and try to refrain the work. If you want to know more, please go to following github site. https://github.com/eriklindernoren/PyTorch-YOLOv3.git.

## Installation
##### Clone and install requirements
    $ git clone https://github.com/Di-Gu/Detection.git
    $ cd Detection
    $ sudo pip3 install -r requirements.txt
    
    Except these three commend lines, user also need to use the newest vision of torchvision. 
    The easiest way is delete the original torchvision in datahub. Then use the following commend 
    to set up the newest vision of tochvision. The git hub link is showing as following.
    https://github.com/pytorch/vision.git
    
    one can install this easily.
    $ pip install torchvision

##### Download pretrained weights (Run this only if you want to train dataset.For test demo, user don't need to type these commends)

    $ cd weights/
    $ bash download_weights.sh
    
## Train

Google colab is suggested to run the 'ece228project.ipynb' file. After you finish set up environment, you can start to training. You can find the dataset in data folder. After setting up the dataset loading path, just run all the files and you can see the training table show up. It may take hours to finish all the epochs.

## Test Demo

After training, you can use the saved '.pth' weight file to run the whole model to see the test result in the same ipynb file. 

# Faster RCNN
The pytorch implementation of Faster RCNN trained on own dataset, The original work is obtained from https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

## Installation
Find pre-trained weights on [here](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), if you want to train from begining.

### Prerequisites
Pytorch 1.0\
CUDA 10.1
### Compilation
```
$ cd FasterRCNN
$ pip install -r requirements 
$ cd lib
$ python setup.py build develop
```

## Train
For this project, trainging is done on Manjaro Linux with GTX 1080 Ti
##### After setup the environment, start training with:
```
$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 4 --lr 4e-3 --lr_decay_step 8 --cuda
```

## Test Demo
Download trained weights [here]()

# C++ based YOLOv3
A rework of the most detailed implementation of YOLOv3 obtained from: https://github.com/AlexeyAB/darknet#requirements.

