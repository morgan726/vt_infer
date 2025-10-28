from ultralytics import YOLO
import cv2

# 模型路径（替换为你的best.pt）
model = YOLO("../yolo/runs/exp2/weights/best.onnx")
# 测试图像路径（替换为你的图像）
img_path = "../img/image.png"

# 预测并绘制结果
results = model(img_path, conf=0.5)  # conf为置信度阈值
annotated_img = results[0].plot()    # 生成带标注的图像

cv2.imwrite("out.png",annotated_img)