import cv2
import numpy as np
import os
import shutil
import random

def adjust_gamma(image, gamma=1.0):
    """调整图像伽马值，gamma<1 调暗，gamma>1 调亮"""
    inv_gamma = 1.0 / gamma
    # 构建伽马校正的查找表
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)  # 应用查找表

def add_gaussian_noise(image, mean=0, sigma=15):
    """添加高斯噪声，sigma控制噪声强度"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype("uint8")  # 确保像素值在0-255之间

def add_salt_pepper_noise(image, prob=0.01):
    """添加椒盐噪声，prob控制噪声比例"""
    output = np.copy(image)
    # 计算盐、椒噪声的像素数量
    num_salt = np.ceil(prob * image.size * 0.5)
    num_pepper = np.ceil(prob * image.size * 0.5)
    
    # 生成盐噪声（白色，像素值255）
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[coords[0], coords[1], coords[2]] = 255
    
    # 生成椒噪声（黑色，像素值0）
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[coords[0], coords[1], coords[2]] = 0
    return output

# ====================== 路径配置 ======================
dir = "/media/dreame/新加卷/economy/dataset/work/kitchen_20251014/"
original_image_dir = dir + "/images"   # 原始图像目录
original_label_dir = dir + "/labels"   # 原始标签目录（YOLO格式txt）
aug_image_dir = dir + "/augmented/images"  # 增强后图像保存目录
aug_label_dir = dir + "/augmented/labels"  # 增强后标签保存目录

# 创建目标目录（若不存在）
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# ====================== 增强参数 ======================
gamma_values = [0.6, 1.5]
noise_options = [
    ("gaussian", lambda img: add_gaussian_noise(img, sigma=25)),
    ("salt_pepper", lambda img: add_salt_pepper_noise(img, prob=0.04))
]

# ====================== 批量处理 ======================
for img_name in os.listdir(original_image_dir):
    # 过滤非图像文件
    if not img_name.endswith((".png", ".jpg", ".jpeg")):
        continue
    img_path = os.path.join(original_image_dir, img_name)
    # 提取图像名（无后缀），用于匹配标签文件
    img_basename = os.path.splitext(img_name)[0]
    label_name = f"{img_basename}.txt"
    label_path = os.path.join(original_label_dir, label_name)

    # 读取原始图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"警告：无法读取图像 {img_path}，跳过")
        continue

    # ---------- 1. 保存原始图像（可选，若需包含原图） ----------
    # cv2.imwrite(os.path.join(aug_image_dir, img_name), img)
    # if os.path.exists(label_path):
    #     shutil.copy(label_path, os.path.join(aug_label_dir, label_name))

    # ---------- 2. 伽马变换增强 ----------
    for gamma in gamma_values:
        gamma_img = adjust_gamma(img, gamma=gamma)
        gamma_img_name = f"gamma{gamma}_{img_name}"
        cv2.imwrite(os.path.join(aug_image_dir, gamma_img_name), gamma_img)
        # 复制对应标签（若存在）
        if os.path.exists(label_path):
            gamma_label_name = f"gamma{gamma}_{label_name}"
            shutil.copy(label_path, os.path.join(aug_label_dir, gamma_label_name))

    # ---------- 3. 随机噪声增强（二选一） ----------
    noise_type, noise_func = random.choice(noise_options)
    noisy_img = noise_func(img)
    noisy_img_name = f"{noise_type}_{img_name}"
    cv2.imwrite(os.path.join(aug_image_dir, noisy_img_name), noisy_img)
    # 复制对应标签（若存在）
    if os.path.exists(label_path):
        noisy_label_name = f"{noise_type}_{label_name}"
        shutil.copy(label_path, os.path.join(aug_label_dir, noisy_label_name))

print("数据增强完成！增强后图像位于 ./augmented/images，标签位于 ./augmented/labels")