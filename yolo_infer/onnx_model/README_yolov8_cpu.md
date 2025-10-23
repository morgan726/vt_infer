#生成cpu后处理的yolov8 onnx模型

##1.下载对应的代码与模型
```
代码
git clone https://github.com/ultralytics/ultralytics.git
git reset --hard d3f097314f9478de7f995d4e4b4ccb0c6fbc65d3
模型
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

##2.导出原始onnx模型
```
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
success = model.export(format="onnx", opset=13)
```
若有报错请检查环境安装包版本，环境所依赖python版本为3.10.12，依赖的软件包版本参考该目录下的requirements_yolov8.txt，可在虚拟环境中搭建环境，搭建命令为
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
deactivte
```