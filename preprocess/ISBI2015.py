import glob
import os
import shutil

import cv2
import numpy as np
import tqdm

root = "data/ISBI_2015"
save = "converted_data/ISBI_2015"

for gt_path in tqdm.tqdm(glob.glob(f"{root}/test*annot")):
    p = gt_path[:-5]
    p_list = p.split(os.path.sep)
    save_name = p_list[-1]
    gts = []
    gt_path_list = glob.glob(os.path.join(gt_path, "Patient-1", "*"))
    for gt_file_path in gt_path_list:
        gts.append(cv2.imread(gt_file_path).sum())

    index = np.argmax(gts, 0)
    for i, gt_file_path in enumerate(gt_path_list[: index + 1][::-1]):
        os.makedirs(os.path.join(save, "gt", save_name + "_former"), exist_ok=True)
        os.makedirs(os.path.join(save, "videos", save_name + "_former"), exist_ok=True)
        shutil.copy(gt_file_path, os.path.join(save, "gt", save_name + "_former", f"{i}.jpg".zfill(10)))
        shutil.copy(
            gt_file_path.replace("annot", "").replace("_mask1", "_flair_pp"),
            os.path.join(save, "videos", save_name + "_former", f"{i}.jpg".zfill(10)),
        )

    for i, gt_file_path in enumerate(gt_path_list[index:]):
        os.makedirs(os.path.join(save, "gt", save_name + "_latter"), exist_ok=True)
        os.makedirs(os.path.join(save, "videos", save_name + "_latter"), exist_ok=True)
        shutil.copy(gt_file_path, os.path.join(save, "gt", save_name + "_latter", f"{i}.jpg".zfill(10)))
        shutil.copy(
            gt_file_path.replace("annot", "").replace("_mask1", "_flair_pp"),
            os.path.join(save, "videos", save_name + "_latter", f"{i}.jpg".zfill(10)),
        )
