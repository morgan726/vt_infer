#生成yolov5 onnx模型

##1.下载对应的代码与模型
代码下载命令
```
git clone -b v7.0 --single-branch https://github.com/ultralytics/yolov5.git
```
模型下载路径
```
https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
```

##2.验证代码功能
环境所依赖python版本为3.10.12，依赖的软件包版本参考该目录下的requirements_yolov5.txt，可在虚拟环境中搭建环境，搭建命令为
```
安装虚拟环境工具
sudo apt install python3-venv
创建虚拟环境
python3.10.12 -m venv venv_yolov5
激活虚拟环境
source venv_yolov5/bin/activate
安装依赖包
pip install --upgrade pip
pip install -r requirements_yolov5.txt
停用虚拟环境
deactivate
```
在yolov5源码路径下运行以下命令，导出原始onnx模型
```
python export.py --weights yolov5s.pt --include onnx
```
若无报错则代表当前环境正常。
##3.导入定制cpu后处理补丁
将0001-yolov5-cpu.patch文件拷贝到yolov5源码路径中，并导入补丁
```
git apply --reject 0001-yolov5-cpu.patch
```

##4.重新导出onnx模型
```
python export.py --weights yolov5s.pt --include onnx --opset 12
```
