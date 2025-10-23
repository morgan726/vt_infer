/**
* @file model_process.cpp
*
* Copyright (C) 2021. Shenshu Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "model_process.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <random>
#include "utils.h"
#include "opencv2/imgproc.hpp"

using namespace std;

static const int BYTE_BIT_NUM = 8; // 1 byte = 8 bit

ModelProcess::ModelProcess()
{
}

ModelProcess::~ModelProcess()
{
    Unload();
    DestroyDesc();
    DestroyInput();
    DestroyOutput();
}

Result ModelProcess::LoadModelFromFileWithMem(const std::string& modelPath)
{
    uint32_t fileSize = 0;
    modelMemPtr_ = Utils::ReadBinFile(modelPath, fileSize);
    modelMemSize_ = fileSize;
    svp_acl_error ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t* >(modelMemPtr_), modelMemSize_, &modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        svp_acl_rt_free(modelMemPtr_);
        ERROR_LOG("load model from file failed, model file is %s", modelPath.c_str());
        return FAILED;
    }

    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath.c_str());
    return SUCCESS;
}

Result ModelProcess::CreateDesc()
{
    modelDesc_ = svp_acl_mdl_create_desc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_get_desc(modelDesc_, modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("get model description failed");
        return FAILED;
    }

    INFO_LOG("create model description success");

    return SUCCESS;
}

void ModelProcess::DestroyDesc()
{
    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }
}

Result ModelProcess::InitInput()
{
    input_ = svp_acl_mdl_create_dataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize, int stride)
{
    svp_acl_data_buffer* inputData = svp_acl_create_data_buffer(inputDataBuffer, bufferSize, stride);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(input_, inputData);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed");
        svp_acl_destroy_data_buffer(inputData);
        inputData = nullptr;
        return FAILED;
    }

    return SUCCESS;
}

size_t ModelProcess::GetInputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_input_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

size_t ModelProcess::GetOutputDataSize(int index) const
{
    svp_acl_data_type dataType = svp_acl_mdl_get_output_data_type(modelDesc_, index);
    return svp_acl_data_type_size(dataType) / BYTE_BIT_NUM;
}

Result ModelProcess::GetInputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const
{
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
        return FAILED;
    }
    stride = svp_acl_mdl_get_input_default_stride(modelDesc_, index);
    if (stride == 0) {
        ERROR_LOG("svp_acl_mdl_get_input_default_stride error!");
        return FAILED;
    }
    bufSize = svp_acl_mdl_get_input_size_by_index(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("svp_acl_mdl_get_input_size_by_index error!");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::GetOutputStrideParam(int index, size_t& bufSize, size_t& stride, svp_acl_mdl_io_dims& dims) const
{
    svp_acl_error ret = svp_acl_mdl_get_output_dims(modelDesc_, index, &dims);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_get_output_dims error!");
        return FAILED;
    }
    stride = svp_acl_mdl_get_output_default_stride(modelDesc_, index);
    if (stride == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_default_stride error!");
        return FAILED;
    }
    bufSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, index);
    if (bufSize == 0) {
        ERROR_LOG("svp_acl_mdl_get_output_size_by_index error!");
        return FAILED;
    }
    return SUCCESS;
}

void ModelProcess::DestroyInput()
{
    if (input_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(input_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(input_, i);
        void* tmp = svp_acl_get_data_buffer_addr(dataBuffer);
        svp_acl_rt_free(tmp);
        svp_acl_destroy_data_buffer(dataBuffer);
    }
    svp_acl_mdl_destroy_dataset(input_);
    input_ = nullptr;
}

Result ModelProcess::CreateOutput()
{
    output_ = svp_acl_mdl_create_dataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }
    size_t outputSize = svp_acl_mdl_get_num_outputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t stride = svp_acl_mdl_get_output_default_stride(modelDesc_, i);
        if (stride == 0) {
            ERROR_LOG("Error, output default stride is %zu.", stride);
            return FAILED;
        }
        size_t bufferSize = svp_acl_mdl_get_output_size_by_index(modelDesc_, i);
        if (bufferSize == 0) {
            ERROR_LOG("Error, output size is %zu.", bufferSize);
            return FAILED;
        }

        void *outputBuffer = nullptr;
        svp_acl_error ret = svp_acl_rt_malloc(&outputBuffer, bufferSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed", bufferSize);
            return FAILED;
        }
        Utils::InitData(static_cast<int8_t*>(outputBuffer), bufferSize);

        svp_acl_data_buffer* outputData = svp_acl_create_data_buffer(outputBuffer, bufferSize, stride);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            return FAILED;
        }
        ret = svp_acl_mdl_add_dataset_buffer(output_, outputData);
        if (ret != SVP_ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed");
            svp_acl_rt_free(outputBuffer);
            svp_acl_destroy_data_buffer(outputData);
            return FAILED;
        }
    }

    INFO_LOG("create model output success");
    return SUCCESS;
}

Result ModelProcess::ClearOutputStrideInvalidBuf(std::vector<int8_t>& buffer, size_t index) const
{
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims dims;
    svp_acl_error ret = GetOutputStrideParam(index, bufSize, bufStride, dims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetOutputStrideParam failed");
        return FAILED;
    }
    if ((bufSize == 0) || (bufStride == 0)) {
        ERROR_LOG("Error, bufSize(%zu) bufStride(%zu) invalid", bufSize, bufStride);
        return FAILED;
    }
    if ((dims.dim_count == 0) || (dims.dims[dims.dim_count - 1] <= 0)) {
        ERROR_LOG("Error, dims para invalid");
        return FAILED;
    }
    int64_t lastDim = dims.dims[dims.dim_count - 1];

    size_t dataSize = GetOutputDataSize(index);
    if (dataSize == 0) {
        ERROR_LOG("Error, dataSize == 0 invalid");
        return FAILED;
    }
    size_t lastDimSize = dataSize * lastDim;
    size_t loopNum = bufSize / bufStride;
    size_t invalidSize = bufStride - lastDimSize;
    if (invalidSize == 0) {
        // not stride invalid space, return directly.
        return SUCCESS;
    }

    for (size_t i = 0; i < loopNum; ++i) {
        size_t offset = bufStride * i + lastDimSize;
        int8_t* ptr = &buffer[offset];
        for (size_t index = 0; index < invalidSize; index++) {
            ptr[index] = 0;
        }
    }
    return SUCCESS;
}

void ModelProcess::WriteOutput(const string& outputFileName, size_t index) const
{
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, index);
    if (dataBuffer == nullptr) {
        ERROR_LOG("output[%zu] dataBuffer nullptr invalid", index);
        return;
    }
    int8_t* outData = (int8_t*)svp_acl_get_data_buffer_addr(dataBuffer);
    size_t outSize = svp_acl_get_data_buffer_size(dataBuffer);
    if (outData == nullptr || outSize == 0) {
        ERROR_LOG("output[%zu] data or size(%zu) invalid", index, outSize);
        return;
    }

    // malloc temp buffer to clear stride useless temp buffer to help output.bin compare
    std::vector<int8_t> tempBuf(outData, outData + outSize);
    if (tempBuf.empty()) {
        ERROR_LOG("tempBuf malloc fail");
        return;
    }

    Result ret = ClearOutputStrideInvalidBuf(tempBuf, index);
    if (ret != SUCCESS) {
        ERROR_LOG("ClearStrideInvalidBuf fail");
        return;
    }

    ofstream fout(outputFileName, ios::out|ios::binary);
    if (fout.good() == false) {
        ERROR_LOG("create output file [%s] failed", outputFileName.c_str());
        return;
    }
    fout.write((char*)&tempBuf[0], tempBuf.size() * sizeof(int8_t));
    fout.close();
    return;
}

void ModelProcess::DumpModelOutputResult() const
{
    stringstream ss;
    size_t outputNum = svp_acl_mdl_get_dataset_num_buffers(output_);
    for (size_t i = 0; i < outputNum; ++i) {
        ss << "output" << executeNum_ << "_" << i << ".bin";
        string outputFileName = ss.str();
        WriteOutput(outputFileName, i);
        ss.str("");
    }
    INFO_LOG("dump data success");
}

enum BoxValue {
    TOP_LEFT_X = 0,
    TOP_LEFT_Y = 1,
    BOTTOM_RIGHT_X = 2,
    BOTTOM_RIGHT_Y = 3,
    SCORE = 4,
    CLASS_ID = 5,
    BBOX_SIZE = 6
};

bool Cmp(const std::vector<float>& veci, const vector<float>& vecj)
{
    if (veci[CLASS_ID] < vecj[CLASS_ID]) {
        return true;
    } else if (veci[CLASS_ID] == vecj[CLASS_ID]) {
        return veci[SCORE] > vecj[SCORE];
    }
    return false;
}

static void PrintResult(const std::vector<std::vector<float>>& boxValue)
{
    if (boxValue.empty()) {
        WARN_LOG("input box empty");
        return;
    }
    std::vector<int> clsNum;
    float cId = boxValue[0][CLASS_ID];
    int validNum = 0;
    for (size_t loop = 0; loop < boxValue.size(); loop++) {
        if (boxValue[loop][CLASS_ID] == cId) {
            validNum++;
        } else {
            clsNum.push_back(validNum);
            cId = boxValue[loop][CLASS_ID];
            validNum = 1;
        }
    }
    clsNum.push_back(validNum);
    int idx = 0;
    int sumNum = 0;
    INFO_LOG("current class valid box number is: %d", clsNum[idx]);
    sumNum += clsNum[idx];
    size_t totalBoxNum = boxValue.size();
    for (size_t loop = 0; loop < totalBoxNum; loop++) {
        if (loop == static_cast<size_t>(sumNum)) {
            idx++;
            INFO_LOG("current class valid box number is: %d", clsNum[idx]);
            sumNum += clsNum[idx];
        }
        INFO_LOG("lx: %lf, ly: %lf, rx: %lf, ry: %lf, score: %lf; class id: %d",
            boxValue[loop][TOP_LEFT_X], boxValue[loop][TOP_LEFT_Y], boxValue[loop][BOTTOM_RIGHT_X],
            boxValue[loop][BOTTOM_RIGHT_Y], boxValue[loop][SCORE], (int)boxValue[loop][CLASS_ID]);
    }
}

static std::string GetDetResultStr(const std::vector<float>& detResult)
{
    enum DET_RESULT_WIDTH {
        INDEX_INTEGER_WIDTH = 4,
        SCORE_INTERGER_WIDTH = 9,
        SCORE_DECIMAL_WIDTH = 8,
        COORD_INTERGER_WIDTH = 4,
        COORD_DECIMAL_WIDTH = 2,
    };
    stringstream ss;
    ss << int(detResult[CLASS_ID]) << "  ";

    ss << std::fixed;

    auto floatStream = [](float val, int width, int precision, stringstream& ss) {
        ss << std::setw(width) << std::setprecision(precision) << val << "  ";
    };

    floatStream(detResult[SCORE], SCORE_INTERGER_WIDTH + SCORE_DECIMAL_WIDTH, SCORE_DECIMAL_WIDTH, ss);
    floatStream(detResult[TOP_LEFT_X], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[TOP_LEFT_Y], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[BOTTOM_RIGHT_X], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detResult[BOTTOM_RIGHT_Y], COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);

    ss << std::endl;
    return ss.str();
}

static void WriteResult(const std::vector<std::vector<float>>& boxValue,
    uint32_t oriImgWidth, uint32_t oriImgHeight, int32_t modelId)
{
    size_t boxSize = boxValue.size();
    if (boxSize == 0) {
        WARN_LOG("input box empty");
        return;
    }
    string tmp = (modelId == static_cast<int>(Yolo::YOLOX)) ? "yolox" : ("yolov" + to_string(modelId));
    tmp = (modelId == static_cast<int>(Yolo::YOLOV5_CPU)) ? "yolov5_cpu" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV8_CPU)) ? "yolov8_cpu" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN)) ? "yolov3_new_rpn" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV5_NEW_RPN)) ? "yolov5_new_rpn" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV7_NEW_RPN)) ? "yolov7_new_rpn" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV8_NEW_RPN)) ? "yolov8_new_rpn" : tmp;
    tmp = (modelId == static_cast<int>(Yolo::YOLOV9_NEW_RPN)) ? "yolov9_new_rpn" : tmp;
    const string fileName = tmp + "_detResult.txt";
    ofstream fout(fileName.c_str());

    std::stringstream ss;
    ss << oriImgWidth << "  " << oriImgHeight << std::endl;
    fout << ss.str();
    ss.str("");
    if (fout.good() == false) {
        ERROR_LOG("fout open fail");
        return;
    }
    for (size_t loop = 0; loop < boxSize; loop++) {
        auto value = boxValue[loop];
        if (value[SCORE] > 1.0f || value[SCORE] < 0.0f) {
            WARN_LOG("invalid score %f", value[SCORE]);
            continue;
        }
        ss << GetDetResultStr(value);
        fout << ss.str();
        ss.str("");
    }
    fout.close();
    return;
}

// yolo/ssd output 0 is num, output 1 is bbox
enum InputOutputId {
    INPUT_IMG_ID = 0,
    OUTPUT_NUM_ID = 0,
    OUTPUT_BBOX_ID = 1
};

const std::vector<std::string> CLASSES = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

static void SaveResult(const std::vector<std::vector<float>>& boxValue, std::vector<Detection>& detections)
{
    for (auto bbox : boxValue) {
        Detection result;
        result.classId = bbox[CLASS_ID];
        result.confidence = bbox[SCORE];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255); // 100: minx value; 255: max value
        result.color = cv::Scalar(dis(gen), dis(gen), dis(gen));

        result.className = CLASSES[result.classId];
        cv::Rect boxs(bbox[TOP_LEFT_X], bbox[TOP_LEFT_Y],
            static_cast<int>(bbox[BOTTOM_RIGHT_X] - bbox[TOP_LEFT_X]),
            static_cast<int>(bbox[BOTTOM_RIGHT_Y] - bbox[TOP_LEFT_Y]));
        result.box = boxs;

        detections.push_back(result);
    }
}

static void DrawResult(const string& imgName, const std::vector<Detection>& detections,
    int32_t imgWidth, int32_t imgHeight, int32_t modelId)
{
    cv::Mat frame = cv::imread(imgName);
    cv::resize(frame, frame, cv::Size(imgWidth, imgHeight));
    for (auto detection : detections) {
        cv::Rect box = detection.box;
        cv::Scalar color = detection.color;

        // Detection box
        cv::rectangle(frame, box, color, 2); // 2: thickness

        // Detection box text,yolov1 doesnt draw className & score
        if (modelId != 1) {
            // 4: len
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0); // 2: thickness
            auto textBoxX = (box.x >= 0) ? box.x : 0;
            auto textBoxY = ((box.y - 40) >= 0) ? (box.y - 40) : box.y;  // 40: offset;
            // 10: offset; 20: offset
            cv::Rect textBox(textBoxX, textBoxY, textSize.width + 10, textSize.height + 20);
            cv::rectangle(frame, textBox, color, cv::FILLED);
            // 40: offset; 10: offset; 30: offset;
            auto textPointY = ((box.y - 40) >= 0) ? (box.y - 10) : (textBoxY + 30);
            cv::putText(frame, classString, cv::Point(textBoxX + 5, textPointY), // 5: offset;
                cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0); // 2: thickness
        }
    }
    string outImgPath = (modelId == static_cast<int>(Yolo::YOLOX)) ?
        "out_img_yolox.jpg" : ("out_img_yolov" + to_string(modelId) + ".jpg");
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV5_CPU)) ? "out_img_yolov5_cpu.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV8_CPU)) ? "out_img_yolov8_cpu.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN)) ? "out_img_yolov3_new_rpn.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV5_NEW_RPN)) ? "out_img_yolov5_new_rpn.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV7_NEW_RPN)) ? "out_img_yolov7_new_rpn.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV8_NEW_RPN)) ? "out_img_yolov8_new_rpn.jpg" : outImgPath;
    outImgPath = (modelId == static_cast<int>(Yolo::YOLOV9_NEW_RPN)) ? "out_img_yolov9_new_rpn.jpg" : outImgPath;
    cv::imwrite(outImgPath, frame);
}

void ModelProcess::OutputModelResult(int32_t modelId, const string& imgName) const
{
    if (modelId == static_cast<int>(Yolo::YOLOX)) {
        return OutputModelResultYoloX(modelId, imgName);
    } else if (modelId == static_cast<int>(Yolo::YOLOV5_CPU)) {
        return OutputModelResultYoloVCpu(modelId, imgName);
    } else if (modelId == static_cast<int>(Yolo::YOLOV8_CPU)) {
        return OutputModelResultYoloV8Cpu(modelId, imgName);
    } else if (modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) || modelId == static_cast<int>(Yolo::YOLOV5_NEW_RPN) ||
        modelId == static_cast<int>(Yolo::YOLOV7_NEW_RPN) || modelId == static_cast<int>(Yolo::YOLOV8_NEW_RPN) ||
        modelId == static_cast<int>(Yolo::YOLOV9_NEW_RPN)) {
        return OutputModelResultNewRpn(modelId, imgName);
    }
    return OutputModelResultYoloV(modelId, imgName);
}

void ModelProcess::OutputModelResultYoloV(int32_t modelId, const string& imgName) const
{
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    // input data shape is nchw, 2 is stand h
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1];
    INFO_LOG("input image width[%d]; height[%d]", imgWidth, imgHeight);

    svp_acl_mdl_io_dims outDims;
    svp_acl_mdl_get_output_dims(modelDesc_, OUTPUT_NUM_ID, &outDims);
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, OUTPUT_NUM_ID);
    auto outData = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBuffer));

    std::vector<Detection> detections {};
    // get valid box number
    std::vector<int> validBoxNum;
    for (uint32_t loop = 0; loop < static_cast<uint32_t>(outDims.dims[outDims.dim_count - 1]); loop++) {
        validBoxNum.push_back(*(outData + loop));
    }
    int totalValidNum = 0;
    for (size_t loop = 0; loop < validBoxNum.size(); loop++) {
        totalValidNum += validBoxNum[loop];
    }
    if (totalValidNum == 0) {
        INFO_LOG("total valid num is zero");
        return;
    }

    // get x y score
    svp_acl_data_buffer* dataBufferValue = svp_acl_mdl_get_dataset_buffer(output_, OUTPUT_BBOX_ID);
    auto outDataValue = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBufferValue));
    svp_acl_mdl_get_output_dims(modelDesc_, OUTPUT_BBOX_ID, &outDims);
    if (outDims.dim_count <= 0) {
        ERROR_LOG("aclrtOutputDims error");
        return;
    }

    size_t wStrideOffset = svp_acl_mdl_get_output_default_stride(modelDesc_, OUTPUT_BBOX_ID) / sizeof(float);
    // box include 6 part which is lx, ly, rx, ry, score, class id
    std::vector<std::vector<float>> bboxes;
    for (int inx = 0; inx < totalValidNum; inx++) {
        std::vector<float> bbox(BBOX_SIZE, 0.0f);
        for (size_t loop = 0; loop < BBOX_SIZE; loop++) {
            bbox[loop] = (*(outDataValue + inx + loop * wStrideOffset));
        }
        bboxes.push_back(bbox);
    }
    std::sort(bboxes.begin(), bboxes.end(), Cmp);
    PrintResult(bboxes);
    WriteResult(bboxes, imgWidth, imgHeight, modelId);
    SaveResult(bboxes, detections);

    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}

static float CalcIou(const vector<float> &box1, const vector<float> &box2)
{
    float area1 = box1[6];
    float area2 = box2[6];
    float xx1 = max(box1[0], box2[0]);
    float yy1 = max(box1[1], box2[1]);
    float xx2 = min(box1[2], box2[2]);
    float yy2 = min(box1[3], box2[3]);
    float w = max(0.0f, xx2 - xx1 + 1);
    float h = max(0.0f, yy2 - yy1 + 1);
    float inter = w * h;
    float ovr = inter / (area1 + area2 - inter);
    return ovr;
}

static void MulticlassNms(vector<vector<float>>& bboxes, const vector<vector<float>>& vaildBox,
    float nmsThr, bool isXywh = true)
{
    const uint8_t scoreIdx = 0;
    const uint8_t xcenterIdx = 1;
    const uint8_t ycenterIdx  = 2; // 2 index
    const uint8_t wIdx = 3; // 3: index
    const uint8_t hIdx = 4; // 4: index
    const uint8_t xminIdx = 1;
    const uint8_t yminIdx  = 2; // 2 index
    const uint8_t xmaxIdx = 3; // 3: index
    const uint8_t ymaxIdx = 4; // 4: index
    const uint8_t classIdIdx = 5; // 5: index
    for (auto &item : vaildBox) { /* score, xcenter, ycenter, w, h, classId */
        float boxXCenter = item[xcenterIdx];
        float boxYCenter = item[ycenterIdx];
        float boxWidth = item[wIdx];;
        float boxHeight = item[hIdx];;

        float x1 = isXywh ? (boxXCenter - boxWidth / 2) : item[xminIdx];
        float y1 = isXywh ? (boxYCenter - boxHeight / 2) : item[yminIdx];
        float x2 = isXywh ? (boxXCenter + boxWidth / 2) : item[xmaxIdx];
        float y2 = isXywh ? (boxYCenter + boxHeight / 2) : item[ymaxIdx];
        float area = (x2 - x1 + 1) * (y2 - y1 + 1);
        bool keep = true;
        /* lx, ly, rx, ry, score, class id, area */
        vector<float> bbox {x1, y1, x2, y2, item[scoreIdx], item[classIdIdx], area};
        for (size_t j = 0; j < bboxes.size(); j++) {
            if (CalcIou(bbox, bboxes[j]) > nmsThr) {
                keep = false;
                break;
            }
        }
        if (keep) {
            bboxes.push_back(bbox);
        }
    }
}
inline static float Sigmod(float a)
{
    return 1.0f / (1.0f + exp(-a));
}

static void GetMaxScoreAndIdx(uint32_t objScoreIdx, uint32_t chnStep, const float* outData,
    float& maxclsSCore, uint32_t& maxClsInx)
{
    uint32_t clsScoreIdx = objScoreIdx + chnStep;
    for (uint32_t c = 0; c < CLASS_NUM; c++) {
        float clsScoreVal = outData[clsScoreIdx];
        if (clsScoreVal > maxclsSCore) {
            maxclsSCore = clsScoreVal;
            maxClsInx = c;
        }
        clsScoreIdx += chnStep;
    }
}

static void ProcessPerDectectionInner(const DetectionInnerParam& innerParam, const vector<float>& gridsX,
    const vector<float>& gridsY, const vector<vector<uint32_t>>& anchorGrids, vector<vector<float>>& vaildBox)
{
    vector<uint32_t> expandedStrides { 32, 16, 8 }; /* 8: 16 : 32: anchor size */
    uint32_t outHeightIdx = innerParam.outHeightIdx;
    uint32_t chnStep = innerParam.chnStep;
    float scoreThr = innerParam.scoreThr;
    float *outData = innerParam.outData;
    size_t wStrideOffset = innerParam.wStrideOffset;
    uint32_t objScoreOffset = innerParam.objScoreOffset;
    for (uint32_t j = 0; j < innerParam.outWidth; j++) {
        for (uint32_t k = 0; k < SCALE_SIZE; k++) {
            uint32_t offset = j + outHeightIdx * wStrideOffset + k * chnStep * OUT_PARM_NUM;
            uint32_t objScoreIdx = offset + objScoreOffset;
            float objScoreVal = Sigmod(outData[objScoreIdx]);
            if (objScoreVal <= scoreThr) {
                continue;
            }
            /* max score */
            float maxclsSCore = 0.0f;
            uint32_t maxClsInx = 0;
            GetMaxScoreAndIdx(objScoreIdx, chnStep, outData, maxclsSCore, maxClsInx);

            float confidenceScore = Sigmod(maxclsSCore) * objScoreVal;
            if (confidenceScore > scoreThr) {
                /* gen box  info */
                uint32_t xCenterIdx = offset;
                uint32_t yCenterIdx = xCenterIdx + chnStep;
                uint32_t boxWidthIdx = yCenterIdx + chnStep;
                uint32_t boxHieghtIdx = boxWidthIdx + chnStep;
                float xCenter = (Sigmod(outData[xCenterIdx]) * 2 + gridsX[j]) *     // 2: alg param
                    expandedStrides[innerParam.detectIdx]; // 2: alg param
                float yCenter = (Sigmod(outData[yCenterIdx]) * 2 + gridsY[outHeightIdx]) *    // 2: alg param
                    expandedStrides[innerParam.detectIdx];
                float tmpValue = Sigmod(outData[boxWidthIdx]) * 2; // 2: alg param
                float boxWidth = tmpValue * tmpValue  * anchorGrids[innerParam.detectIdx][(k << 1)];
                tmpValue = Sigmod(outData[boxHieghtIdx]) * 2; // 2: alg param
                float boxHieght = tmpValue * tmpValue  * anchorGrids[innerParam.detectIdx][(k << 1) +1];

                vaildBox.push_back({confidenceScore, xCenter, yCenter, boxWidth,
                    boxHieght, static_cast<float>(maxClsInx)});
            }
        }
    }
}
void ModelProcess::ProcessPerDectection(size_t detectIdx, vector<vector<float>>& vaildBox) const
{
    svp_acl_mdl_io_dims outDims;
    svp_acl_mdl_get_output_dims(modelDesc_, detectIdx, &outDims);
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, detectIdx);
    DetectionInnerParam innerParam;
    innerParam.scoreThr = scoreThr_;
    innerParam.detectIdx = detectIdx;
    innerParam.outData = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBuffer));

    innerParam.wStrideOffset = svp_acl_mdl_get_output_default_stride(modelDesc_, detectIdx) / sizeof(float);
    uint32_t outHeight = outDims.dims[outDims.dim_count - 2];
    innerParam.outWidth = outDims.dims[outDims.dim_count - 1];
    innerParam.chnStep = outHeight * innerParam.wStrideOffset;
    vector<uint32_t> expandedStrides { 32, 16, 8 }; /* 8: 16 : 32: anchor size */
    vector<uint32_t> hSizes{ 20, 40, 80 }; // imgh / expandedStrides
    vector<uint32_t> wSizes{ 20, 40, 80 }; // imgw / expandedStrides
    vector<vector<uint32_t>> anchorGrids {
        {116, 90, 156, 198, 373, 326}, // p5/32
        {30, 61, 62, 45, 59, 119}, // p4/16
        {10, 13, 16, 30, 33, 23}  // p3/8
    };

    /* gen grids */
    vector<float>gridsX(wSizes[detectIdx]);
    vector<float>gridsY(hSizes[detectIdx]);
    for (uint32_t i = 0; i < hSizes[detectIdx]; i++) {
        gridsY[i] = i - 0.5; // 0.5: alg param
    }
    for (uint32_t i = 0; i < wSizes[detectIdx]; i++) {
        gridsX[i] = i - 0.5; // 0.5: alg param
    }
    innerParam.objScoreOffset = 4 * innerParam.chnStep; // 4: offset
    for (uint32_t i = 0; i < outHeight; i++) {
        innerParam.outHeightIdx = i;
        ProcessPerDectectionInner(innerParam, gridsX, gridsY, anchorGrids, vaildBox);
    }
}

void ModelProcess::FilterYolov5Box(vector<vector<float>>& vaildBox) const
{
    size_t detectionOutNum = svp_acl_mdl_get_num_outputs(modelDesc_);
    /* gen box */
    for (size_t n = 0; n < detectionOutNum; n++) {
        ProcessPerDectection(n, vaildBox);
    }
}

void ModelProcess::OutputModelResultNewRpn(int32_t modelId, const string& imgName) const
{
    // do bbox by small ops
    // npu return filterVectorVal filterVectorNum filterVectorIdx filterVectorNum coord&classId
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    // input data shape is nchw, 2 is stand h
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1];

    std::vector<Detection> detections {};
    const uint8_t outNumIdx = modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) ? 1 : 2;
    svp_acl_data_buffer* outNumBuffer = svp_acl_mdl_get_dataset_buffer(output_, outNumIdx);
    float outNum = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(outNumBuffer))[0];
    INFO_LOG("input image width[%d] height[%d], npu outBoxNum = %f", imgWidth, imgHeight, outNum);

    const uint8_t scoreValIdx = modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) ? 0 : 1;
    svp_acl_data_buffer* scoreBuffer = svp_acl_mdl_get_dataset_buffer(output_, scoreValIdx);
    auto scoreData = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(scoreBuffer));

    const uint8_t scoreIndexIdx = modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) ? 2 : 3;
    svp_acl_data_buffer* maxScoreIdxBuffer = svp_acl_mdl_get_dataset_buffer(output_, scoreIndexIdx);
    auto maxScoreIdx = reinterpret_cast<uint16_t*>(svp_acl_get_data_buffer_addr(maxScoreIdxBuffer));

    const uint8_t coordIdx = modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) ? 4 : 0;
    svp_acl_data_buffer* coordBuffer = svp_acl_mdl_get_dataset_buffer(output_, coordIdx);
    auto coord = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(coordBuffer));
    size_t coordStrideOffset = svp_acl_mdl_get_output_default_stride(modelDesc_, coordIdx) / sizeof(float);

    const uint8_t maxClassIdx = modelId == static_cast<int>(Yolo::YOLOV3_NEW_RPN) ? 3 : 4;
    svp_acl_data_buffer* maxClassBuffer = svp_acl_mdl_get_dataset_buffer(output_, maxClassIdx);
    auto maxClass = reinterpret_cast<uint16_t*>(svp_acl_get_data_buffer_addr(maxClassBuffer));

    std::vector<std::vector<float>> bboxes;
    for (uint32_t idx = 0; idx < outNum; idx++) {
        uint32_t coordIdx = maxScoreIdx[idx];
        // bbox:score xmin ymin xmax ymax classId
        std::vector<float> bbox{scoreData[idx], coord[coordIdx], coord[coordIdx + 1 * coordStrideOffset],
            coord[coordIdx + 2 * coordStrideOffset], coord[coordIdx + 3 * coordStrideOffset],
            static_cast<float>(maxClass[coordIdx])};
        bboxes.push_back(bbox);
    }

    std::sort(bboxes.begin(), bboxes.end(), [](const vector<float>& veci, const vector<float>& vecj) {
        return veci[0] > vecj[0];
    });

    std::vector<std::vector<float>> finalBoxes;
    MulticlassNms(finalBoxes, bboxes, nmsThr_, false);
    PrintResult(finalBoxes);
    WriteResult(finalBoxes, imgWidth, imgHeight, modelId);
    SaveResult(finalBoxes, detections);
    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}

void ModelProcess::OutputModelResultYoloVCpu(int32_t modelId, const string& imgName) const
{
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    // input data shape is nchw, 2 is stand h
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1];
    INFO_LOG("input image width[%d]; height[%d]", imgWidth, imgHeight);

    vector<vector<float>> vaildBox; /* score, xcenter, ycenter, w, h, classId */
    /* gen box */
    FilterYolov5Box(vaildBox);
    std::sort(vaildBox.begin(), vaildBox.end(), [](const vector<float>& veci, const vector<float>& vecj) {
        if (veci[0] > vecj[0]) {
            return true;
        }
        return false;
    });

    // box include 6 part which is lx, ly, rx, ry, score, class id
    vector<vector<float>> bboxes;
    MulticlassNms(bboxes, vaildBox, nmsThr_);
    if (bboxes.size() == 0) {
        INFO_LOG("total valid num is zero");
        return;
    }

    vector<Detection> detections {};
    std::sort(bboxes.begin(), bboxes.end(), Cmp);
    PrintResult(bboxes);
    WriteResult(bboxes, imgWidth, imgHeight, modelId);
    SaveResult(bboxes, detections);

    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}

void ModelProcess::FilterYolov8Box(int32_t modelId, vector<vector<float>>& vaildBox) const
{
    svp_acl_mdl_io_dims outDims;
    svp_acl_error ret = svp_acl_mdl_get_output_dims(modelDesc_, OUTPUT_NUM_ID, &outDims);
    if (ret != SVP_ACL_SUCCESS || outDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_output_dims error!");
    }

    int outWidth = outDims.dims[outDims.dim_count - 1];
    auto stride = svp_acl_mdl_get_output_default_stride(modelDesc_, 0) / sizeof(float);
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, 0);
    auto xCenter = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBuffer));
    auto yCenter = xCenter + stride;
    auto boxWidth = yCenter + stride;
    auto boxHeight = boxWidth + stride;
    auto classScore = boxHeight + stride;

    for (int j = 0; j < outWidth; j++) {
        float maxSore = 0.f;
        uint8_t maxClassId = 0;
        auto tmpClassScore = classScore;
        for (uint8_t classId = 0; classId < CLASS_NUM; classId++) {
            if (*tmpClassScore > maxSore) {
                maxClassId = classId;
                maxSore = *tmpClassScore;
            }
            tmpClassScore += stride;
        }
        if (maxSore > scoreThr_) {
            vaildBox.push_back({maxSore, xCenter[j], yCenter[j], boxWidth[j],
                boxHeight[j], static_cast<float>(maxClassId)});
        }
        classScore++;
    }
}

void ModelProcess::OutputModelResultYoloV8Cpu(int32_t modelId, const string& imgName) const
{
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    // input data shape is nchw, 2 is stand h
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1];
    INFO_LOG("input image width[%d]; height[%d]", imgWidth, imgHeight);

    vector<vector<float>> vaildBox; /* score, xcenter, ycenter, w, h, classId */
    FilterYolov8Box(modelId, vaildBox);
    std::sort(vaildBox.begin(), vaildBox.end(), [](const vector<float>& veci, const vector<float>& vecj) {
        if (veci[0] > vecj[0]) {
            return true;
        }
        return false;
    });

    // box include 6 part which is lx, ly, rx, ry, score, class id:
    vector<vector<float>> bboxes;
    MulticlassNms(bboxes, vaildBox, nmsThr_);
    if (bboxes.size() == 0) {
        INFO_LOG("total valid num is zero");
        return;
    }
    vector<Detection> detections {};
    std::sort(bboxes.begin(), bboxes.end(), Cmp);
    PrintResult(bboxes);
    WriteResult(bboxes, imgWidth, imgHeight, modelId);
    SaveResult(bboxes, detections);

    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}

static void GenYoloxBoxInfo(const svp_acl_mdl_io_dims& inDims, float* outData, size_t wStrideOffset, uint32_t outHeight)
{
    // input data shape is nchw, 2 is stand h
    uint32_t imgHeight = static_cast<uint32_t>(inDims.dims[inDims.dim_count - 2]);
    uint32_t imgWidth = static_cast<uint32_t>(inDims.dims[inDims.dim_count - 1]);
    vector<uint32_t> expandedStrides { 8, 16, 32 }; /* 8: 16 : 32: anchor size */
    vector<uint32_t> hSizes(SCALE_SIZE);
    vector<uint32_t> wSizes(SCALE_SIZE);
    for (uint32_t i = 0; i < expandedStrides.size(); i++) {
        hSizes[i] = imgHeight / expandedStrides[i]; // 80 , 40, 20
        wSizes[i] = imgWidth / expandedStrides[i]; // 80 , 40, 20
    }
    vector<uint32_t> gridsX(outHeight);
    vector<uint32_t> gridsY(outHeight);
    uint32_t gridsCnt = 0;
    for (uint32_t k = 0; k < SCALE_SIZE; k++) {
        for (uint32_t i = 0; i < hSizes[k]; i++) {
            for (uint32_t j = 0; j < wSizes[k]; j++) {
                gridsX[gridsCnt] = j;
                gridsY[gridsCnt] = i;
                gridsCnt++;
            }
        }
    }

    float *xCenter = outData;
    float *yCenter = xCenter + 1;
    float *boxW = yCenter + 1;
    float *boxH = boxW + 1;

    uint32_t expandedIdxPhase0 = hSizes[0] * wSizes[0]; // 6400
    uint32_t expandedIdxPhase1 = hSizes[1] * wSizes[1] + expandedIdxPhase0; // 6400 + 1600
    uint32_t expandedStrideIdx = 0;
    for (uint32_t i = 0; i < outHeight; i++) {
        if (i >= expandedIdxPhase0) {
            expandedStrideIdx = 1;
        }
        if (i >= expandedIdxPhase1) {
            expandedStrideIdx = 2; // 2:
        }
        *xCenter = (*xCenter + gridsX[i]) * expandedStrides[expandedStrideIdx];
        *yCenter = (*yCenter + gridsY[i]) * expandedStrides[expandedStrideIdx];
        *boxW = exp(*boxW) * expandedStrides[expandedStrideIdx];
        *boxH = exp(*boxH) * expandedStrides[expandedStrideIdx];
        xCenter += wStrideOffset;
        yCenter += wStrideOffset;
        boxW += wStrideOffset;
        boxH += wStrideOffset;
    }
}

void ModelProcess::FilterBbox(const float *outData, size_t wStrideOffset, uint32_t outHeight,
    vector<vector<float>>& bboxes) const
{
    vector<vector<float>> vaildBox; /* score, xcenter, ycenter, w, h, classId */
    const float *objScore = outData + 4; /* 4: x, y, w, h */

    float tmpScore = 0.0;
    for (uint32_t i = 0; i < outHeight; i++) {
        const float *classScore = objScore + 1;
        float maxScore = 0.0;
        uint32_t maxClsIdx = CLASS_NUM;
        bool validScore = false;
        for (uint32_t j = 0; j < CLASS_NUM; j++) {
            tmpScore = (*objScore) * (*classScore);
            classScore++;
            if (tmpScore <= scoreThr_) {
                continue;
            }
            if (tmpScore > maxScore) {
                maxClsIdx = j;
                maxScore = tmpScore;
            }
            validScore = true;
        }

        if (validScore) {
            const  float *boxInfo = outData + i * wStrideOffset;
            float xCenter = *boxInfo;
            float yCenter = *(boxInfo + 1);
            float boxWidth = *(boxInfo + 2);
            float boxHeight = *(boxInfo + 3);
            vaildBox.push_back({maxScore, xCenter, yCenter, boxWidth, boxHeight, static_cast<float>(maxClsIdx)});
        }
        objScore += wStrideOffset;
    }

    std::sort(vaildBox.begin(), vaildBox.end(), [](const vector<float>& veci, const vector<float>& vecj) {
        if (veci[0] > vecj[0]) {
            return true;
        }
        return false;
    });

    MulticlassNms(bboxes, vaildBox, nmsThr_);
}
void ModelProcess::OutputModelResultYoloX(int32_t modelId, const string& imgName) const
{
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = svp_acl_mdl_get_input_dims(modelDesc_, INPUT_IMG_ID, &inDims);
    if (ret != SVP_ACL_SUCCESS || inDims.dim_count <= 2) { // 2: dim count
        ERROR_LOG("svp_acl_mdl_get_input_dims error!");
    }
    // input data shape is nchw, 2 is stand h
    int imgHeight = inDims.dims[inDims.dim_count - 2];
    int imgWidth = inDims.dims[inDims.dim_count - 1]; /* 85: x, y , w, h, objscore, classScore(80) */
    INFO_LOG("input image width[%d]; height[%d]", imgWidth, imgHeight);

    svp_acl_mdl_io_dims outDims;
    svp_acl_mdl_get_output_dims(modelDesc_, OUTPUT_NUM_ID, &outDims);
    svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, OUTPUT_NUM_ID);
    auto outData = reinterpret_cast<float*>(svp_acl_get_data_buffer_addr(dataBuffer));
    size_t wStrideOffset = svp_acl_mdl_get_output_default_stride(modelDesc_, 0) / sizeof(float);
    int outHeight = outDims.dims[outDims.dim_count - 2]; // 8400

    GenYoloxBoxInfo(inDims, outData, wStrideOffset, outHeight);

    // box include 6 part which is lx, ly, rx, ry, score, class id
    vector<vector<float>> bboxes;
    FilterBbox(outData, wStrideOffset, outHeight, bboxes);
    if (bboxes.size() == 0) {
        INFO_LOG("total valid num is zero");
        return;
    }
    std::sort(bboxes.begin(), bboxes.end(), Cmp);
    PrintResult(bboxes);
    WriteResult(bboxes, imgWidth, imgHeight, modelId);
    vector<Detection> detections {};
    SaveResult(bboxes, detections);

    DrawResult(imgName, detections, imgWidth, imgHeight, modelId);
    INFO_LOG("output data success");
    return;
}
void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(output_); ++i) {
        svp_acl_data_buffer* dataBuffer = svp_acl_mdl_get_dataset_buffer(output_, i);
        void* data = svp_acl_get_data_buffer_addr(dataBuffer);
        (void)svp_acl_rt_free(data);
        (void)svp_acl_destroy_data_buffer(dataBuffer);
    }

    (void)svp_acl_mdl_destroy_dataset(output_);
    output_ = nullptr;
}

Result ModelProcess::Execute()
{
    svp_acl_error ret = svp_acl_mdl_execute(modelId_, input_, output_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u", modelId_);
        return FAILED;
    }
    executeNum_++;
    INFO_LOG("model execute success");
    return SUCCESS;
}

Result ModelProcess::CreateBuf(int index)
{
    void *bufPtr = nullptr;
    size_t bufSize = 0;
    size_t bufStride = 0;
    svp_acl_mdl_io_dims inDims;
    svp_acl_error ret = GetInputStrideParam(index, bufSize, bufStride, inDims);
    if (ret != SUCCESS) {
        ERROR_LOG("Error, GetInputStrideParam failed");
        return FAILED;
    }

    ret = svp_acl_rt_malloc(&bufPtr, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed. size is %zu", bufSize);
        return FAILED;
    }
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufSize);

    ret = CreateInput(bufPtr, bufSize, bufStride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(bufPtr);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateInputBuf(const string& filePath)
{
    size_t devSize = 0;
    size_t stride = 0;
    svp_acl_mdl_io_dims inputDims;
    // only support single input model
    Result ret = GetInputStrideParam(0, devSize, stride, inputDims);
    if (ret != SUCCESS) {
        ERROR_LOG("GetStrideParam error");
        return FAILED;
    }
    size_t dataSize = GetInputDataSize(0);
    if (dataSize == 0) {
        ERROR_LOG("GetInputDataSize == 0 error");
        return FAILED;
    }
    void *picDevBuffer = Utils::GetDeviceBufferOfFile(filePath, inputDims, stride, dataSize);
    if (picDevBuffer == nullptr) {
        ERROR_LOG("get pic device buffer failed");
        return FAILED;
    }

    ret = InitInput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute InitInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }

    ret = CreateInput(picDevBuffer, devSize, stride);
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateInput failed");
        svp_acl_rt_free(picDevBuffer);
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetDetParas(int32_t modelId)
{
    vector<float> detPara;
    string rpnFile = "../src/yolov" + to_string(modelId) + "_rpn.txt";
    Result ret = Utils::ReadFloatFile(rpnFile, detPara);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("ReadRpnFile error");
        return FAILED;
    }
    // 4: nms, score, minheight, minwidth
    if (detPara.size() != 4) {
        ERROR_LOG("detPara.size %zu should be 4", detPara.size());
        return FAILED;
    }
    ret = SetDetParas(detPara);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("SetDetParas error");
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::SetDetParas(const vector<float>& detPara)
{
    enum DetParaEnum {
        NMS_THR = 0,
        SCORE_THR = 1,
        MIN_HEIGHT = 2,
        MIN_WIDTH = 3,
    };

    void *bufPtr = nullptr;
    size_t bufferSize = sizeof(float) * detPara.size();
    svp_acl_error ret = svp_acl_rt_malloc(&bufPtr, bufferSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("malloc device buffer failed");
        return FAILED;
    }
    Utils::InitData(static_cast<int8_t*>(bufPtr), bufferSize);

    auto bufFloat = reinterpret_cast<float*>(bufPtr);
    bufFloat[NMS_THR] = detPara[NMS_THR];
    bufFloat[SCORE_THR] = detPara[SCORE_THR];
    bufFloat[MIN_HEIGHT] = detPara[MIN_HEIGHT];
    bufFloat[MIN_WIDTH] = detPara[MIN_WIDTH];

    // det para is 4 * sizeof(float) = 16 = default stride
    svp_acl_data_buffer* inputData = svp_acl_create_data_buffer(bufPtr, bufferSize, bufferSize);
    if (inputData == nullptr) {
        (void)svp_acl_rt_free(bufPtr);
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }

    ret = svp_acl_mdl_add_dataset_buffer(input_, inputData);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed");
        (void)svp_acl_rt_free(bufPtr);
        (void)svp_acl_destroy_data_buffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    return SUCCESS;
}

Result ModelProcess::CreateTaskBufAndWorkBuf()
{
    // 2 is stand taskbuf and workbuf
    if (svp_acl_mdl_get_num_inputs(modelDesc_) <= 2) {
        ERROR_LOG("input dataset Num is error.");
        return FAILED;
    }
    size_t datasetSize = svp_acl_mdl_get_dataset_num_buffers(input_);
    if (datasetSize == 0) {
        ERROR_LOG("input dataset Num is 0.");
        return FAILED;
    }
    for (size_t loop = datasetSize; loop < svp_acl_mdl_get_num_inputs(modelDesc_); loop++) {
        Result ret = CreateBuf(loop);
        if (ret != SUCCESS) {
            ERROR_LOG("execute Create taskBuffer and workBuffer failed");
            return FAILED;
        }
    }
    return SUCCESS;
}

void ModelProcess::Unload()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    svp_acl_error ret = svp_acl_mdl_unload(modelId_);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u", modelId_);
    }

    if (modelDesc_ != nullptr) {
        (void)svp_acl_mdl_destroy_desc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelMemPtr_ != nullptr) {
        svp_acl_rt_free(modelMemPtr_);
        modelMemPtr_ = nullptr;
        modelMemSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        svp_acl_rt_free(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
}

Result ModelProcess::SetYolov3BboxParas()
{
    for (uint32_t i = 1; i <= 3; i++) { // 3: 3 bbox input
        vector<float> anchors;
        string dataPath = "../data/bbox" + to_string(i) + "_data.txt";
        printf("dataPath = %s\n", dataPath.c_str());
        Result ret = Utils::ReadFloatFile(dataPath, anchors);
        size_t devSize = 0;
        size_t stride = 0;
        svp_acl_mdl_io_dims inputDims;
        ret = GetInputStrideParam(i, devSize, stride, inputDims);
        if (ret != SUCCESS) {
            ERROR_LOG("GetStrideParam error");
            return FAILED;
        }

        void* binFileBufferData = nullptr;
        svp_acl_error aclRet = svp_acl_rt_malloc(&binFileBufferData, devSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);

        int64_t inputH = inputDims.dims[2]; // 2:axis h
        int64_t inputW = inputDims.dims[3]; // 3:axis w
        for (uint32_t i = 0; i < inputH; i++) {
            for (uint32_t j = 0; j < inputW; j++) {
                static_cast<float*>(binFileBufferData)[i * stride / sizeof(float) + j] = anchors[i * inputW + j ];
            }
        }

        svp_acl_data_buffer* inputData = svp_acl_create_data_buffer(binFileBufferData, devSize, stride);
        if (inputData == nullptr) {
            ERROR_LOG("can't create data buffer, create input failed");
            return FAILED;
        }

        aclRet = svp_acl_mdl_add_dataset_buffer(input_, inputData);
        if (aclRet != SVP_ACL_SUCCESS) {
            ERROR_LOG("add input dataset buffer failed");
            svp_acl_destroy_data_buffer(inputData);
            inputData = nullptr;
            return FAILED;
        }
    }
    return SUCCESS;
}
