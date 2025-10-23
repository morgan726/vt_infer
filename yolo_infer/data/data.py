import os
import random

# 1. 核心参数配置（可根据需求调整）
target_dir = "/media/dreame/新加卷/economy/dataset/work/kitchen_20251014"  # 待遍历的文件夹
output_file = "data.txt"  # 输出100张图像路径的文件
img_extensions = (".png", ".jpg", ".jpeg")  # 支持的图像格式
target_count = 10000  # 目标筛选数量

# 2. 路径合法性检查
if not os.path.exists(target_dir):
    print(f"错误：目标路径不存在 → {target_dir}")
    exit(1)

# 3. 第一步：收集所有符合格式的图像绝对路径
all_img_paths = []
for root, dirs, files in os.walk(target_dir):
    for file_name in files:
        # 只保留指定格式的图像文件（不区分大小写，如.PNG/.JPG也能识别）
        if file_name.lower().endswith(img_extensions):
            file_path = os.path.join(root, file_name)
            absolute_path = os.path.abspath(file_path)
            all_img_paths.append(absolute_path)

# 4. 第二步：检查图像总数，处理“不足100张”的情况
total_img = len(all_img_paths)
if total_img == 0:
    print(f"错误：目标路径下未找到任何 {img_extensions} 格式的图像")
    exit(1)
elif total_img < target_count:
    print(f"提示：目标路径下仅找到 {total_img} 张图像，将全部写入（不足100张）")
    selected_paths = all_img_paths  # 不足时直接取全部
else:
    # 随机筛选100张，设置random.seed确保每次筛选结果可复现（如需不同结果可删除seed行）
    random.seed(42)  # 固定种子：每次运行选同一批；删除此行则每次随机
    selected_paths = random.sample(all_img_paths, target_count)  # 无重复随机挑选

# 5. 第三步：将筛选后的路径写入文件
with open(output_file, "w", encoding="utf-8") as f:
    for path in selected_paths:
        f.write(path + "\n")

# 6. 输出结果汇总
print("=" * 50)
print(f"处理完成！")
print(f"目标路径：{target_dir}")
print(f"筛选图像总数：{len(selected_paths)} 张")
print(f"输出文件路径：{os.path.abspath(output_file)}")
print(f"包含格式：{img_extensions}")
print("=" * 50)