#生成带定制rpn算子的yolov8 onnx模型

##1.下载对应的代码与模型
代码下载命令
```
git clone https://github.com/ultralytics/ultralytics.git
```
在源码路径中运行命令
```
git reset --hard d3f097314f9478de7f995d4e4b4ccb0c6fbc65d3
```
模型下载路径
```
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

##2.验证代码功能
环境所依赖python版本为3.10.12，依赖的软件包版本参考该目录下的requirements_yolov8.txt，可在虚拟环境中搭建环境，搭建命令为
```
安装虚拟环境工具
sudo apt install python3-venv
创建虚拟环境
python3.10.12 -m venv venv_yolov8
激活虚拟环境
source venv_yolov8/bin/activate
安装依赖包
pip install --upgrade pip
pip install -r requirements_yolov8.txt
停用虚拟环境
deactivate
```
在yolov8源码路径下运行以下命令，导出原始onnx模型
```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
success = model.export(format="onnx", opset=13)
```
若无报错则代表当前环境正常。
##3.导入定制rpn算子补丁或者导入定制new_rpn算子补丁
将0001-yolov8-rpn.patch文件拷贝到yolov8源码路径中，并导入补丁
```
git apply --reject 0001-yolov8-rpn.patch
```
若需要使用新版本的rpn后处理，导入补丁的命令如下
```
git apply --reject 0001-yolov8-new-rpn.patch
```
##4.重新导出onnx模型
```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
success = model.export(format="onnx", opset=13)
```
对于新版本rpn后处理，需把生成的yolov8n.onnx文件重命名为yolov8n_new_rpn.onnx