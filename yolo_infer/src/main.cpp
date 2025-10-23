/**
* @file main.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <iostream>
#include <map>
#include <string>
#include "sample_process.h"
#include "utils.h"

using namespace std;

static void LogInfoContext(void)
{
    INFO_LOG("./main param\nparam is model and input image bin pair(default 1)");
    INFO_LOG("param 3: yolov3.om and dog_bike_car_yolov3.bin");
    INFO_LOG("param 5: yolov5.om and dog_bike_car_yolov5.bin");
    INFO_LOG("param 5_cpu: yolov5_cpu.om and dog_bike_car_yolov5.bin");
    INFO_LOG("param 7: yolov7.om and dog_bike_car_yolov7.bin");
    INFO_LOG("param 8: yolov8.om and dog_bike_car_yolov8.bin");
    INFO_LOG("param 8_cpu: yolov8_cpu.om and dog_bike_car_yolov8.bin");
    INFO_LOG("param x: yolox.om and dog_bike_car_yolox.bin");
    INFO_LOG("param 3_new_rpn: yolov3_new_rpn.om and dog_bike_car_yolov3.bin");
    INFO_LOG("param 5_new_rpn: yolov5_new_rpn.om and dog_bike_car_yolov5.bin");
    INFO_LOG("param 7_new_rpn: yolov7_new_rpn.om and dog_bike_car_yolov7.bin");
    INFO_LOG("param 8_new_rpn: yolov8_new_rpn.om and dog_bike_car_yolov8.bin");
    INFO_LOG("param 9_new_rpn: yolov8_new_rpn.om and dog_bike_car_yolov9.bin");
}

int GetInputId(string tmp)
{
    int inputId = 1;
    if (tmp[0] == 'x') {
        inputId = static_cast<int>(Yolo::YOLOX);
    } else if (tmp == "5_cpu") {
        inputId = static_cast<int>(Yolo::YOLOV5_CPU);
    } else if (tmp == "8_cpu") {
        inputId = static_cast<int>(Yolo::YOLOV8_CPU);
    } else if (tmp == "3_new_rpn") {
        inputId = static_cast<int>(Yolo::YOLOV3_NEW_RPN);
    } else if (tmp == "5_new_rpn") {
        inputId = static_cast<int>(Yolo::YOLOV5_NEW_RPN);
    } else if (tmp == "7_new_rpn") {
        inputId = static_cast<int>(Yolo::YOLOV7_NEW_RPN);
    } else if (tmp == "8_new_rpn") {
        inputId = static_cast<int>(Yolo::YOLOV8_NEW_RPN);
    } else if (tmp == "9_new_rpn") {
        inputId = static_cast<int>(Yolo::YOLOV9_NEW_RPN);
    } else {
        inputId = stoi(tmp);
    }
    return inputId;
}

int main(int argc, char *argv[])
{
    LogInfoContext();
    int modelOpt = 1;
    if (argc > 1) {
        string tmp(argv[1]);
        int inputId = GetInputId(tmp);
        switch (inputId) {
            case 3: // 3: yolov3
            case 5: // 5: yolov5
            case 7: // 7: yolov7
            case 8: // 8: yolov8
            case 10: // 10:x yolox
            case static_cast<int>(Yolo::YOLOV5_CPU):
            case static_cast<int>(Yolo::YOLOV8_CPU):
            case static_cast<int>(Yolo::YOLOV3_NEW_RPN):
            case static_cast<int>(Yolo::YOLOV5_NEW_RPN):
            case static_cast<int>(Yolo::YOLOV7_NEW_RPN):
            case static_cast<int>(Yolo::YOLOV8_NEW_RPN):
            case static_cast<int>(Yolo::YOLOV9_NEW_RPN):
                modelOpt = inputId;
                break;
            default:
                ERROR_LOG("option invalid %d", modelOpt);
                return FAILED;
        }
        INFO_LOG("yolo v%s", argv[1]);
    }
    SampleProcess sampleProcess(modelOpt);
    Result ret = sampleProcess.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }

    ret = sampleProcess.Process();
    if (ret != SUCCESS) {
        ERROR_LOG("sample process failed");
        return FAILED;
    }

    INFO_LOG("execute sample success");
    return SUCCESS;
}
