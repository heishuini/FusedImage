import os
import shutil

# 原始文件夹路径
src_folder = "../data/DIV2K_Flickr_LSDIR5000/gt_1000"
# 目标文件夹路径
dst_folder = "../data/DIV2K_Flickr_LSDIR5000/gt_1000name"

os.makedirs(dst_folder, exist_ok=True)

for i in range(1, 1001):
    old_name = os.path.join(src_folder, f"{i:04d}.png")
    new_name = os.path.join(dst_folder, f"lq_{i:06d}.png")
    if os.path.exists(old_name):
        shutil.copy(old_name, new_name)  # 或者用 shutil.move(old_name, new_name) 来移动
        print(f"Copied: {old_name} -> {new_name}")
    else:
        print(f"File not found: {old_name}")
