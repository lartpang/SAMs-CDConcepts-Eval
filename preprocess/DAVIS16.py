import os
import shutil

data_list = "F:/1-SemanticSegmentation/DAVIS-data/DAVIS/ImageSets/1080p/val.txt"
data_root = "F:/1-SemanticSegmentation/DAVIS-data/DAVIS"
save_root = "F:/1-SemanticSegmentation/DAVIS-data/DAVIS/DAVIS16-Val"

for line in open(data_list, mode="r", encoding="utf-8"):
    image_sub_path, mask_sub_path = line.strip().split()
    image_sub_path = image_sub_path[1:]
    mask_sub_path = mask_sub_path[1:]
    image_path = os.path.join(data_root, image_sub_path)
    mask_paht = os.path.join(data_root, mask_sub_path)

    new_image_path = os.path.join(save_root, image_sub_path)
    os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
    shutil.copy(image_path, new_image_path)

    new_mask_path = os.path.join(save_root, mask_sub_path)
    os.makedirs(os.path.dirname(new_mask_path), exist_ok=True)
    shutil.copy(mask_paht, new_mask_path)
