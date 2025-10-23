# 基于Caffe/ONNX yolo 网络实现图片目标检测

## 功能描述

该样例主要是基于yolo (V3,V5,V7,V8,V9,X)网络实现图片目标检测的功能，其中yoloV3为caffe模型，yoloV5,V7,V8,V9,X为onnx模型
该样例上板运行时存在占用系统内存较多，可能存在oom killer异常的情况，建议扩大系统内存为512M。

yolo系列网络原始框架 [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
yolov5网络源码 [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
yolov7网络源码 [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
yolov8网络源码 [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
yolov9网络源码 [https://github.com/WongKinYiu/yolov9/](https://github.com/WongKinYiu/yolov9/)
yoloX网络原始框架 [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

在该样例中：

1.  先使用样例提供的脚本 transferPic.py，将*.bmp或*.jpg图片(暂不支持.png格式)都转换为*.bin格式，同时将图片分辨率缩放为模型输入所需分辨率。

2.  加载离线模型om文件，对图片进行同步推理，得到目标检测结果的置信度和框偏移量预测值，再对推理结果进行目标检测后处理，得到目标检测框。

    在加载离线模型前，需提前将Caffe/Onnx模型文件转换为om离线模型。

## 原理介绍

在该Sample中，涉及的关键功能点，如下所示：

-   **初始化**

    -   调用svp_acl_init接口初始化ACL配置。
    -   调用svp_acl_finalize接口实现ACL去初始化。

-   **Device管理**

    -   调用svp_acl_rt_set_device接口指定用于运算的Device。
    -   调用svp_acl_rt_get_run_mode接口获取运行模式，根据运行模式的不同，内部处理流程不同。
    -   调用svp_acl_rt_reset_device接口复位当前运算的Device，回收Device上的资源。

-   **Context管理**

    -   调用svp_acl_rt_create_context接口创建Context。
    -   调用svp_acl_rt_destroy_context接口销毁Context。

-   **Stream管理**

    -   调用svp_acl_rt_create_stream接口创建Stream。
    -   调用svp_acl_rt_destroy_stream接口销毁Stream。

-   **内存管理**

    -   调用svp_acl_rt_malloc接口申请Device上的内存。
    -   调用svp_acl_rt_free接口释放Device上的内存。

-   **模型推理**

    -   调用svp_acl_mdl_load_from_mem接口从\*.om文件加载模型。
    -   调用svp_acl_mdl_execute接口执行模型推理，同步接口。
    -   调用svp_acl_mdl_unload接口卸载模型。

-   **数据后处理**

    提供样例代码，处理模型推理的结果。

    另外，样例中提供了自定义接口DumpModelOutputResult，用于将模型推理的结果写入文件（运行可执行文件后，推理结果文件在运行环境上的应用可执行文件的同级目录下），默认未调用该接口，用户可在sample_process.cpp中，在调用OutputModelResult接口前，增加如下代码调用DumpModelOutputResult接口：

    ```
    // use function DumpModelOutputResult
    // if want to dump output result to file in the current directory
    modelProcess.DumpModelOutputResult();
    modelProcess.OutputModelResult();
    ```

## 目录结构

样例代码结构如下所示。

```
├── data
│   ├── ...            //测试数据
├── inc
│   ├── model_process.h               //声明模型处理相关函数的头文件
│   ├── sample_process.h              //声明资源初始化/销毁相关函数的头文件
│   ├── utils.h                       //声明公共函数（例如：文件读取函数）的头文件
├── script
│   ├── transferPic.py               //将*.jpg转换为*.bin
│   ├── drawbox.py               //画框脚本
├── src
│   ├── acl.json         //系统初始化的配置文件
│   ├── CMakeLists.txt         //编译脚本
│   ├── main.cpp               //主函数，图片分类功能的实现文件
│   ├── model_process.cpp      //模型处理相关函数的实现文件
│   ├── sample_process.cpp     //资源初始化/销毁相关函数的实现文件
│   ├── utils.cpp              //公共函数（例如：文件读取函数）的实现文件
│   ├── *_rpn.txt            //硬化加速配置
├── caffe_model
│   ├── *.caffemodel    //模型权重文件
│   ├── *.prototxt    //模型描述文件
├── onnx_model
│   ├── *.onnx  //onnx模型
│   ├── README_*.md  //onnx模型生成方法
│   ├── 0001-*.patch  //onnx模型生成依赖补丁
├── .project     //工程信息文件，包含工程类型、工程描述、运行目标设备类型等
├── CMakeLists.txt    //编译脚本，调用src目录下的CMakeLists文件
├── *.json            //Mindstudio工具转换模型配置文件
├── insert_op.cfg        //yolov(3/5/7/8) aipp配置文件
├── insert_op_yolox.cfg  //yolox aipp配置文件

```

## 环境要求

-   操作系统及架构：Ubuntu 22.04 x86\_64、 arm
-   编译器：
    -   Hi3516CV610 形态编译器：g++ 或 arm-v01c02-linux-musleabi/arm-v01c02-linux-gnueabi
-   芯片：Hi3516CV610
-   python及依赖的库：python3.10.12、Pillow库
-   已在环境上部署智能软件栈

## 配置环境变量

-   **Hi3516CV610：**

执行setenv.sh脚本配置环境变量。$\{install\_path\}表示开发套件包Ascend-toolkit所在的路径。

```
source ${install_path}/svp_latest/x86_64-linux/script/setenv.sh
```

或按以下步骤配置环境变量

1.  开发环境上，设置模型转换依赖的环境变量。

    $\{install\_path\}表示开发套件包Ascend-toolkit所在的路径。

    ```
    export PATH=${install_path}/svp_latest/atc/bin:$PATH
    ```

2.  开发环境上，设置环境变量，编译脚本src/CMakeLists.txt通过环境变量所设置的头文件、库文件的路径来编译代码。
    如下为设置环境变量的示例，请将$HOME/Ascend/ascend-toolkit/svp_latest替换为开发套件包Ascend-toolkit下对应架构的acllib的路径。

    - 当运行环境操作系统架构为x86时，执行以下命令：

    ```
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/x86_64-linux
    ```

    - 当运行环境操作系统架构为Arm时，执行以下命令：

    ```
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib_linux.x86_64
    ```

       使用“$HOME/Ascend/ascend-toolkit/svp_latest/acllib/lib32/stub”目录下的\*.so库，是为了编译基于ACL接口的代码逻辑时，不依赖其它组件（例如Driver）的任何\*.so库。编译通过后，在Host上运行应用时，会根据环境变量LD\_LIBRARY\_PATH链接到“acllib/lib32“目录下的\*.so库，并自动链接到依赖其它组件的\*.so库。

3.  运行环境上，设置环境变量，运行应用时需要根据环境变量找到对应的库文件。

    - 若运行环境上安装的是开发套件包Ascend-toolkit，环境变量设置如下：

    ```
    export LD_LIBRARY_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/lib32/stub
    ```

## 编译运行

1.  模型转换。
    1.  以运行用户登录开发环境。
    2.  设置环境变量。
        $\{install\_path\}表示开发套件包Ascend-toolkit的安装路径。
        ```
        export LD_LIBRARY_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/acllib/lib32/stub
        ```

    3.  将网络转换为适配智能处理器的离线模型适配SoC的离线模型（\*.om文件）。
        切换到样例目录，执行如下命令
        对于caffe模型，以yolov3为例：

        ```
        atc  --dump_data=0 --input_shape="data:1,3,416,416" --input_type="data:UINT8" --log_level=0 --weight="./caffe_model/yolov3.caffemodel" --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov3" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=0 --compile_mode=0 --save_original_model=true --model="./caffe_model/yolov3.prototxt" --image_list="data:./data/image_ref_list.txt"
        ```

        对于onnx模型，以yolov5为例：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov5" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov5s.onnx" --image_list="images:./data/image_ref_list.txt"
        ```
        相较于Hi3519DV500，受硬件限制，rpn硬化算子filter、sort、nms通过aicpu算子实现，对性能有一定影响。

        对于yolov5网络, cpu后处理：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov5_cpu" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov5s_cpu.onnx" --image_list="images:./data/image_ref_list.txt"
        ```

        对于yolov5 a8w4网络, cpu后处理：

        ```
        atc --dump_data=0 --input_shape="x:1,3,640,640" --input_type="x:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov5_cpu" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov5s_a8w4_deploy_model.onnx" --image_list="x:./data/image_ref_list.txt" --gfpq_param_file="./data/yolov5s_a8w4_quant_param_record.txt"
        ```

        对于onnx模型，yolov8 cpu后处理, 由于网络最后一层为concat层，一路为坐标信息，一路为分值信息，避免量化导致结果不对，需配置分组量化系数， 使该层不做量化，量化系数配置文件可参考./onnx_model目录下的calibration_param.txt：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov8_cpu" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --gfpq_param_file=./onnx_model/calibration_param.txt --save_original_model=true --model="./onnx_model/yolov8n_cpu.onnx" --image_list="images:./data/image_ref_list.txt"
        ```

        对于yolox网络：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolox" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op_yolox.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolox_s.onnx" --image_list="images:./data/image_ref_list.txt"
        ```

        对于新版本rpn的yolov3网络：

        ```
        atc --dump_data=0 --input_shape="data:1,3,416,416" --input_type="data:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov3_new_rpn" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=0 --compile_mode=1 --save_original_model=true --weight="./caffe_model/yolov3.caffemodel" --model="./caffe_model/yolov3_new_rpn.prototxt" --image_list="data:./data/image_ref_list.txt;data_bbox1:./data/bbox1_data.txt;data_bbox2:./data/bbox2_data.txt;data_bbox3:./data/bbox3_data.txt" --out_nodes="argmaxIdxs:0" --output_type="argmaxIdxs:0:U16"
        ```

        对于新版本rpn的yolov5网络：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov5_new_rpn" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov5s_new_rpn.onnx" --image_list="images:./data/image_ref_list.txt" --out_nodes="/model.24/Concat_5:0" --output_type="/model.24/Concat_5:0:U16"
        ```

        对于新版本rpn的yolov7网络：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov7_new_rpn" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov7-tiny_new_rpn.onnx" --image_list="images:./data/image_ref_list.txt" --out_nodes="/model.77/Concat_5:0" --output_type="/model.77/Concat_5:0:U16"
        ```

        对于新版本rpn的yolov8网络：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov8_new_rpn" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov8n_new_rpn.onnx" --image_list="images:./data/image_ref_list.txt" --out_nodes="/model.22/ArgMax:0" --output_type="/model.22/ArgMax:0:U16"
        ```

        对于新版本rpn的yolov9网络：

        ```
        atc --dump_data=0 --input_shape="images:1,3,640,640" --input_type="images:UINT8" --log_level=0 --online_model_type=0 --batch_num=1 --input_format=NCHW --output="./model/yolov9_new_rpn" --soc_version=Hi3516CV610 --insert_op_conf=./insert_op.cfg --framework=5 --compile_mode=0 --save_original_model=true --model="./onnx_model/yolov9-m-converted.onnx" --image_list="images:./data/image_ref_list.txt" --out_nodes="/model.22/ArgMax:0" --output_type="/model.22/ArgMax:0:U16"
        ```
        -   --model：原始模型文件路径。
        -   --weight：权重文件路径。
        -   --framework：原始框架类型。0：表示Caffe；5：表示ONNX; 6：ABSTRACT（使用构图接口）。
        -   --insert_op_conf: 静态aipp配置参数。
        -   --soc\_version：此处配置为Hi3516CV610。
        -   --input\_format：输入数据的Format。
        -   --output：生成的om文件存放在“样例目录/model“目录下。建议使用命令中的默认设置，否则在编译代码前，您还需要修改sample_process.cpp中的omModelPath参数值。
        -   --gfpq_param_file:配置分组量化参数配置文件

        ```
        const string omModelPath = "../model/yolo(v3/v5/v5_cpu/v7/v8/v8_cpu/x/v5_new_rpn/v7_new_rpn/v8_new_rpn/v9_new_rpn)_original.om";
        ```

2.  编译代码。
    1.  需要将sample发布包内的third_party目录连同sample一起拷贝到开发环境，third_party目录下为opencv的交叉编译库。
    2.  切换到样例目录，创建目录用于存放编译文件，例如，本文中，创建的目录为“build“。

        ```
        mkdir -p build
        ```

    3.  切换到“build“目录，执行**cmake**生成编译文件。

        “../src“表示CMakeLists.txt文件所在的目录，请根据实际目录层级修改。

        - 当开发环境与运行环境操作系统架构相同时，执行如下命令编译程序。

          指令仿真：
          ```
          cd build
          cmake ../src -Dtarget=Simulator_Instruction -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
          ```

          功能仿真：

          ```
          cd build
          cmake ../src -Dtarget=Simulator_Function -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
          ```

        - 当开发环境与运行环境操作系统架构不同时，执行以下命令进行交叉编译。

          例如，当开发环境为X86架构，运行环境为ARM架构时，执行以下命令进行交叉编译。其中交叉编译器有arm-v01c02-linux-musleabi-gcc/arm-v01c02-linux-gnueabi-gcc可供选择使用。

          ```
          cd build
          cmake ../src -Dtarget=board -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=arm-v01c02-linux-musleabi-gcc
          ```

    4.  执行**make**命令，生成的可执行文件main在“样例目录/out“目录下。

        ```
        make
        cd -
        mv ../../out ./
        ```

    该样例同时提供了编译脚本，切换到样例目录，直接执行命令：
    ```
    ./build.sh
    ```

    同时在out目录下生成功能仿真可执行文件（func_main），指令仿真可执行文件（inst_main）和板端可执行文件（board_glibc_main/board_musl_main）；board_musl_main由arm-v01c02-linux-musleabi编译链编译，board_glibc_main由arm-v01c02-linux-gnueabi编译链编译。

3.  准备输入图片。
    1.  以运行用户登录开发环境。
    2.  切换到“样例目录/data”目录下，执行yolov3/yolov5/yolov5_cpu/yolov7/yolov8/yolox模型的transferPic.py脚本，将*.bmp或*.jpg转换为*.bin，同时将图片分辨率缩放为模型输入所需的640\*640。在“样例目录/data”目录下生成*.bin文件, yolov5_cpu可以和yolov5共用*.bin。
    ```
    python3.10.12 ../script/transferPic.py 3 //yolov3
    python3.10.12 ../script/transferPic.py 5 //yolov5
    python3.10.12 ../script/transferPic.py 5 //yolov5_cpu
    python3.10.12 ../script/transferPic.py 7 //yolov7
    python3.10.12 ../script/transferPic.py 8 //yolov8
    python3.10.12 ../script/transferPic.py 8 //yolov8_cpu
    python3.10.12 ../script/transferPic.py 9 //yolov9
    python3.10.12 ../script/transferPic.py x //yolox
    ```
    如果执行脚本报错“ModuleNotFoundError: No module named 'PIL'”，则表示缺少Pillow库，请使用**pip3.10.12 install Pillow --user**命令安装Pillow库。

4.  运行应用。
    1.  以运行用户将开发环境的样例目录及目录下的文件上传到运行环境（Host），例如“$HOME/samples”。

    2.  由于yolo sample支持使用opencv直接画框且保存图片结果，所以
    a. 当开发环境与运行环境操作系统架构不同时，需要将opencv的依赖库拷贝到运行环境的/lib目录，或者用户自行设置环境变量用于链接lib库，例如：

       ```
       方式1：cp samples/third_party/musl/opencv/lib/* /lib/
       方式2：export LD_LIBRARY_PATH=$BOARD_PATH/samples/third_party/musl/opencv/lib/:$LD_LIBRARY_PATH
       ```
    b. 当开发环境与运行环境操作系统架构相同时(仿真运行环境)，需要用户自行设置环境变量用于链接lib库，例如：
       ```
       export LD_LIBRARY_PATH=$HOME/Ascend/ascend-toolkit/svp_latest/atc/third_party_lib/:$LD_LIBRARY_PATH
       ```

       另外，不同编译编译工具链的运行环境需要拷贝或者链接对应的依赖库，依赖库路径如下：

       ```
       1、arm-v01c02-linux-musleabi-gcc 编译工具链，即运行环境的/lib下带有musl字样，如：ld-musl-arm.so.1，发布包路径为：
       samples/third_party/musl/opencv/lib
       2、arm-v01c02-linux-gnueabi-gcc 编译工具链，即运行环境的/lib下带有linux字样，如：ld-linux-arm.so.1，发布包路径为：
       samples/third_party/glibc/opencv/lib
       3、g++ 编译工具链，即仿真运行环境，发布包路径为：
       Ascend/ascend-toolkit/svp_latest/atc/third_party_lib
       ```

    3. 以运行用户登录运行环境（Host）。

    4. 切换到可执行文件main所在的目录，例如“$HOME/samples/out”，给该目录下的main文件加执行权限。

    ```
    chmod +x main
    ```

    5. 切换到可执行文件main所在的目录，例如“$HOME/samples/out”，运行可执行文件。

    ```
    ./main 3 //yolov3
    ./main 3_new_rpn //yolov3 new rpn
    ./main 5 //yolov5
    ./main 5_cpu //yolov5 cpu
    ./main 5_new_rpn //yolov5 new rpn
    ./main 7 //yolov7
    ./main 7_new_rpn //yolov7 new rpn
    ./main 8 //yolov8
    ./main 8_cpu //yolov8 cpu
    ./main 8_new_rpn //yolov8 new rpn
    ./main 9_new_rpn //yolov9 new rpn
    ./main x //yolox
    ```

5. 输出后处理
    本例中，模型执行后，基于推理结果，提供两种画框方式，详见下一章节。

## 模型画框脚本使用

1. 方式1：

   ./main X 执行完后，在out目录下会直接保存名为 “out_img_yolovX.jpg”。(X为具体yolo的版本号，比如1,3,5等)。

2. 方式2：

   1. 由后处理输出检测结果并保存至xxx_detResult.txt中，其中第一行为输入图片的W和H的大小，下面每一行有6个值，分别是对应框的classId, score, leftX, leftY, rightX, rightY。

   2. 切换到样例目录“/out”目录下，执行drawbox.py脚本。

      ```
      python3.10.12 ../script/drawbox.py -i ../data/dog_bike_car.jpg -t yolo(v3/v5/v7/v8/x/v5_cpu/v8_cpu/v5_new_rpn/v7_new_rpn/v8_new_rpn/v9_new_rpn)_detResult.txt
      ```

## 后处理阈值配置
1.对于yolov3、v5、v7、v8、v9网络，支持检测网硬化加速，目标检测网络相关离线参数需在caffe prototxt和onnx模型中配置，目标检测过滤相关在线参数（如filter阈值，nms阈值等）通过float文件读入，顺序依次为nms、score、minheight、minwidth，通过rpn.txt配置。具体细节请参考《ATC工具使用指南》。

2.对于yolox和yolov5、yolov7 cpu后处理网络，其后处理通过cpu实现，其filter、nms阈值为 model\_process.h 中的成员变量scoreThr\_ 与nmsThr\_，可通过修改其默认值修改来实现阈值配置。

3.对于yolov3、v5、v7、v8、v9 新版本rpn网络，其filter通过npu实现，参数在模型中配置。对于caffe模型，需修改prototxt中filterVal与filterIdx的filter\_thresh值；对于onnx模型，需修改pytorch代码中创建FilterVector自定义算子filter\_val、filter\_idx时配置的filter\_thresh值，需注意，filter\_val、filter\_idx算子的filter\_thresh参数必须配为相同值。nms处理通过cpu实现。nms阈值为 model\_process.h 中的成员变量nmsThr\_，可通过修改其默认值修改来实现阈值配置。


## 附录：Bbox计算公式
用于对齐Bbox计算流程。
1.yolov3
```
x方向（y方向计算流程与x方向类似）：
cX = (sigmoid(x) + col) / gridNumWidth
halfW = exp(w) * bias / gridNumWidth * 0.5
pMinX = halfW - cX
pMaxX = halfW + cX

score：
finalScore = sigmoid(objScore) * max(sigmoid(classScore))
```
2.yolov5/yolov7
```
x方向（y方向计算流程与x方向类似）：
cX = (2 * sigmoid(x) - 0.5 + col) / gridNumWidth
halfW = sigmoid(w) * sigmoid(w) * bias / gridNumWidth * 0.5
pMinX = halfW – cX
pMaxX = halfW + cX

score：
finalScore = sigmoid(objScore) * max(sigmoid(classScore))
```
3.yolov8/yolov9
```
坐标计算部分由bbox前的算子实现，参考yolov8源码，采用DFL损失函数。
score在bbox中计算，计算公式如下：
finalScore = max(sigmoid(classScore))
```
4.yolox
```
x方向（y方向计算流程与x方向类似）：
xCenter = cx + tx
w = exp(tw)
score：
finalScore =objScore * classScore
```
其中col为列索引，gridNumWidth为锚点宽，x和w为上一层的输出，bias为锚点bias参数，cX为检测框中心点x坐标，halfW为检测框宽*0.5，pMinX为检测框Xmin，pMaxX为检测框Xmax。objScore和classScore为上一层的输出，finalScore为最终分数。
