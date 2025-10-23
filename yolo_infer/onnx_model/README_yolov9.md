#生成带定制rpn算子的yolov9 onnx模型

##1.下载对应的代码与模型
代码下载命令
```
git clone https://github.com/WongKinYiu/yolov9.git
```
在源码路径中运行命令
```
git reset --hard 5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff
```
模型下载路径
```
https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-m-converted.pt
```

##2.验证代码功能
环境所依赖python版本为3.10.12，依赖的软件包版本参考该目录下的requirements_yolov9.txt，可在虚拟环境中搭建环境，搭建命令为
```
安装虚拟环境工具
sudo apt install python3-venv
创建虚拟环境
python3.10.12 -m venv venv_yolov9
激活虚拟环境
source venv_yolov9/bin/activate
安装依赖包
pip install --upgrade pip
pip install -r requirements_yolov9.txt
停用虚拟环境
deactivate
```
在yolov9源码路径下运行以下命令，导出原始onnx模型
```
python export.py --weights=yolov9-m-converted.pt --include onnx
```
若无报错则代表当前环境正常。
##3.导入定制rpn算子补丁
将0001-yolov9-new-rpn.patch文件拷贝到yolov9源码路径中，并导入补丁
```
git apply --reject 0001-yolov9-new-rpn.patch
```

##4.重新导出onnx模型
```
python export.py --weights=yolov9-m-converted.pt --include onnx
```