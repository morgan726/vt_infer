#生成带定制rpn算子的yolov7 onnx模型

##1.下载对应的代码与模型
代码下载命令
```
git clone -b v0.1 --single-branch https://github.com/WongKinYiu/yolov7.git
```
模型下载路径
```
https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
```

##2.验证代码功能
环境所依赖python版本为3.10.12，依赖的软件包版本参考该目录下的requirements_yolov7.txt，可在虚拟环境中搭建环境，搭建命令为
```
安装虚拟环境工具
sudo apt install python3-venv
创建虚拟环境
python3.10.12 -m venv venv_yolov7
激活虚拟环境
source venv_yolov7/bin/activate
安装依赖包
pip install --upgrade pip
pip install -r requirements_yolov7.txt
停用虚拟环境
deactivate
```
在yolov7源码路径下运行以下命令，导出原始onnx模型
```
python models/export.py --weights ./yolov7-tiny.pt --grid --img-size 640 640
```
若无报错则代表当前环境正常。

##3.导入定制rpn算子补丁或者导入定制new_rpn算子补丁
将0001-yolov7-rpn.patch文件拷贝到yolov7源码路径中，并导入补丁
```
git apply --reject 0001-yolov7-rpn.patch
```
若需要使用新版本的rpn后处理，导入补丁的命令如下
```
git apply --reject 0001-yolov7-new-rpn.patch
```
##4.重新导出onnx模型
```
python models/export.py --weights ./yolov7-tiny.pt --grid --img-size 640 640
```
对于新版本rpn后处理，需把生成的yolov7-tiny.onnx文件重命名为yolov7-tiny_new_rpn.onnx