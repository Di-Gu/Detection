# Marine Animal Detection
This repository is built for UCSD ECE 228 final project, maintained by Di Gu, Shuangcheng Yang, Yixun Zhang and Kunmao Li. Special thanks to [OBSEA](https://obsea.es/) for sharing their [underwater camera image]() with us.\
Please refer this [paper]() for more details on our project.

## File Organization

- FasterRCNN - a pytorch version of Faster RCNN
- Results - detection results from models
- YOLOv3 - a pytorch version of YOLOv3
- data - training dataset with cfg files for YOLOv3
- tools - codes for data processing, visualization

## Model 1: PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation. This file is a reversion of yolov3 and most of the py files are coming from another github author. We test and try to refrain the work. If you want to know more, please go to this [repo](https://github.com/eriklindernoren/PyTorch-YOLOv3.git).

### Installation
##### Clone and install requirements
    $ git clone https://github.com/Di-Gu/Detection.git
    $ cd Detection/YOLOv3
    $ sudo pip3 install -r requirements.txt
    
    Except these three commend lines, user also need to use the newest vision of torchvision. 
    The easiest way is delete the original torchvision in datahub. Then use the following commend 
    to set up the newest vision of tochvision. The git hub link is showing as following.
    https://github.com/pytorch/vision.git
    
    one can install this easily.
    $ pip install torchvision

##### Download pretrained weights (Run this only if you want to train dataset. For test demo, user don't need to type these commands)

    $ cd weights/
    $ bash download_weights.sh
    
### Train

Google colab is suggested to run the 'ece228project.ipynb' file. After you finish set up environment, you can start to training. You can find the dataset in data folder. After setting up the dataset loading path, just run all the files and you can see the training table show up. It may take hours to finish all the epochs.

### Test Demo

After training, you can use the saved '.pth' weight file to run the whole model to see the test result in the same ipynb file. 

## Model 2: Faster RCNN
The pytorch implementation of Faster RCNN trained on own dataset, The original work is obtained from [Jianwei Yang](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0).

### Installation
Find pre-trained weights on [here](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), if you want to train from begining.

#### Prerequisites
Pytorch 1.0\
CUDA 10.1
#### Compilation
```
$ cd FasterRCNN
$ pip install -r requirements 
$ cd lib
$ python setup.py build develop
```

### Train
For this project, trainging is done on Manjaro Linux with GTX 1080 Ti
##### After setup the environment, start training with:
```
$ CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net res101 --bs 4 --lr 4e-3 --lr_decay_step 8 --cuda
```

### Test Demo
Download trained weights [here](https://drive.google.com/file/d/141dOq4E_IOPE25SH5X5Zy8ssAsMTJ1BQ/view?usp=sharing). Then run following line for testing
```
$ python demo.py --checksession 1 --checkepoch 20 --checkpoint 139 --cuda --load_dir models
```

## Model 3: C++ based YOLOv3
A rework of the most detailed implementation of YOLOv3 obtained from [AlexeyAB](https://github.com/AlexeyAB/darknet).

### Installation
Following the step from the [repo](https://github.com/AlexeyAB/darknet#requirements)\
After successfully compiled on Windows\
Download the pretrained weight for the convolutional layers from [here](http://pjreddie.com/media/files/darknet53.conv.74)\
Relocate the cfg files in the data folder accordingly

### Train
On Windows powershell run
```
.\darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map 2>1 | tee train_yolov3.log
```

### Test Demo
Download trained weights [here](https://drive.google.com/file/d/1YHSIXxkbrUSm8JiZSFOEEXJZfDBscMu_/view?usp=sharing). Then run following line for testing
```
darknet.exe detector test data/obj.data yolo-obj-test.cfg backup/yolo-obj_last.weights data/test/test_3.jpg -thresh 0.5
```
