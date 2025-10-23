from ultralytics import YOLO
import os

# ====================== 配置参数 ======================
# 模型选择：使用预训练权重（n/s/m/l/x，n为最小模型，速度快；x为最大模型，精度高）
model_name = "yolov8n.pt"  # 可替换为yolov8s.pt、yolov8m.pt等

# 数据集配置文件路径（指向你的kitchen.yaml）
data_path = os.path.join("../python", "coco.yaml")

# 训练参数
epochs = 100  # 训练轮数（根据数据集大小调整，小数据集可设50-100，大数据集200+）
batch_size = 48  # 批次大小（根据GPU显存调整，显存不足则减小，如8、4）
img_size = 640  # 输入图像尺寸（YOLO默认640x640）
device = 0  # 训练设备：0表示第1块GPU，-1表示CPU（建议用GPU，否则训练很慢）
project = "runs"  # 训练结果保存的项目文件夹
name = "exp"  # 本次实验名称（会在project下生成exp文件夹）
workers = 4  # 数据加载线程数（根据CPU核心数调整）

# ====================== 开始训练 ======================
if __name__ == "__main__":
    # 加载YOLOv8模型（预训练权重）
    model = YOLO(model_name)
    
    # 开始训练
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        workers=workers
    )
    
    # 训练完成后，在验证集上评估模型（train已包含评估，这里可选）
    metrics = model.val()
    print("验证集评估结果：")
    print(f"mAP@0.5: {metrics.box.map50:.3f}")  # mAP@0.5指标
    print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")  # mAP@0.5:0.95指标

    # 保存最终模型（默认已保存在project/name/weights/best.pt）
    best_model_path = os.path.join(project, name, "weights", "best.pt")
    print(f"最佳模型已保存至：{best_model_path}")