# Detection

```
.\darknet.exe detector train data/obj.data yolo-obj.cfg darknet53.conv.74 -map 2>1 | tee train_yolov3.log
```

```
darknet.exe detector test data/obj.data yolo-obj-test.cfg backup/yolo-obj_46000.weights data/test/test_3.jpg -thresh 0.5
```
