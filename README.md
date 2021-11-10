# yolov5s-involution
A simple project fusing involution operator into yolov5s, and provide tensorrt support.
## Reference
* [TensorrtX](https://github.com/wang-xinyu/tensorrtx) provide  the implementation of yolov5 with TensorRT API.
* [Involution](https://github.com/d-li14/involution) is the offical implementation of involution and the [Issue44](https://github.com/d-li14/involution/issues/44) provide Fast and generic implementation using OpenMP and CUDA.
* [Unfold-Tensorrt](https://github.com/grimoire/amirstan_plugin) provide the unfold operator with tensorrtAPI.
* [Yolov5](https://github.com/ultralytics/yolov5) provide the framework of Yolov5.
## Quick Start
1 clone our project to your JetsonTX2

* `git clone https://github.com/entropyfeng/yolov5s-involution.git`

2 build our project and the more detail guidance can be achive from [this](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5). 
```
cd {projectPath}
mkdir build
cd build
cmake ..
make
```
3 if your want to check the inference time in your only device
```
# build engine file
yolov5 -s {your WTS file path such as yolov5s-A-voc.wts in this project} {path of to build engine file} s A/B/C
# if you want to build the engine of original yolov5
yolov5s {your WTS file path}  {path of to build engine file} s 
# test the inference time of current model in current device.
yolov5 -dtest {path of engine file} in
# if you want to test the original yolov5
yolov5 -dtest {path of engine file}
```
4 if you want to transfer the weight of your training, you can clone [Yolov5](https://github.com/ultralytics/yolov5) into your training environment, and after your complete current training, you can transfering `gen_wts.py` to your yolov5 project,  calling this python file and transfer the `.wts` file into your edge device.
## Performance In Pascal VOC
We conduct experiment on JetsonTX2, it's worth noting that the inference time and accuracy with some fluctuation and affected including but not limited to temperature of current device, scheduling policy of CPU or GPU, power supply and other tasks being performed.
|  Model   | mAP  |  mAP50   |GPU forward time in TX2 with TensorRT |GPU forward time in TX2 without TensorRT |
|  ----  | ----  |  ----  | ----  |----  |
| Yolov5s  | 0.439 |0.651  | 25.5 |54.3|
| Yolov5s-A  | 0.456 |0.658  | 32.1 |68.1|
| Yolov5s-B  | 0.458 |0.663  | 72.6 |144.1|
| Yolov5s-C  | 0.438 |0.657  | 65.9 |129.8|

Yolov5s-A seems to have a higher cost performance ratio.
## Tips
* The inference time measured on Jetson Nano may take large fluctuation compare to measured on Jetson TX2  Through multiple measurements. That is, the inference time relatively stable on Jetson Tx2.
* The SiLu activation function is superior to ReLu in part datasets, you can test the accuracy of both activation functions separately on your own dataset.(This step is no need to test on your edge device, test on your train environment is more convenience).
* In our own experiments, the default TensorRT version in Jetson TX2 is 7, and the version in Jetson Nano is 8, which cause different API definition.
