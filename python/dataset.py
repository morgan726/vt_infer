import os
import random

# 配置路径（根目录、原始图像/标签、增强图像/标签）
root_dir = "/media/dreame/新加卷/economy/dataset/work/kitchen_20251014/"
original_img_dir = os.path.join(root_dir, "images")       # 原始图像目录
augmented_img_dir = os.path.join(root_dir, "augmented", "images")  # 增强图像目录
output_dir = root_dir  # txt文件保存目录（当前根目录）

# 图像格式过滤
img_extensions = (".png", ".jpg", ".jpeg")

original_imgs = []
for img in os.listdir(original_img_dir):
    if img.lower().endswith(img_extensions):
        # 相对路径格式：images/文件名（相对于root_dir）
        rel_path = os.path.join(original_img_dir, img).replace("\\", "/")
        print(rel_path)
        original_imgs.append(rel_path)

augmented_imgs = []
for img in os.listdir(augmented_img_dir):
    if img.lower().endswith(img_extensions):
        # 相对路径格式：augmented/images/文件名（相对于root_dir）
        rel_path = os.path.join(augmented_img_dir, img).replace("\\", "/")
        augmented_imgs.append(rel_path)

# 3. 合并原始和增强图像，打乱顺序
all_imgs = original_imgs + augmented_imgs
random.seed(42)  # 固定随机种子，确保划分一致
random.shuffle(all_imgs)

# 4. 8:2划分训练集和验证集
split_idx = int(len(all_imgs) * 0.8)
train_imgs = all_imgs[:split_idx]  # 训练集（80%）
val_imgs = all_imgs[split_idx:]    # 验证集（20%）

# 5. 生成train.txt（包含原始和增强图像的训练集路径）
with open(os.path.join(output_dir, "train.txt"), "w") as f:
    for img_path in train_imgs:
        f.write(f"{img_path}\n")

# 6. 生成val.txt（包含原始和增强图像的验证集路径）
with open(os.path.join(output_dir, "val.txt"), "w") as f:
    for img_path in val_imgs:
        f.write(f"{img_path}\n")

print(f"生成完成：\n"
      f"总图像数：{len(all_imgs)} 张（原始+增强）\n"
      f"训练集：{len(train_imgs)} 张 → {output_dir}/train.txt\n"
      f"验证集：{len(val_imgs)} 张 → {output_dir}/val.txt")