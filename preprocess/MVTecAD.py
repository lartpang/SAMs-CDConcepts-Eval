import glob
import os
import shutil

import cv2
import numpy as np
import tqdm

root = "data/mvtec"
save = "converted_data/mvtec"
os.makedirs(os.path.join(save, "images"), exist_ok=True)
os.makedirs(os.path.join(save, "gt"), exist_ok=True)

for p in tqdm.tqdm(glob.glob(f"{root}/*/test/*/*.png")):
    gt_path = p.replace("test", "ground_truth")[:-4] + "_mask.png"
    p_list = p.split(os.path.sep)
    save_name = p_list[-4] + "_" + p_list[-2] + "_" + p_list[-1]
    shutil.copy(p, os.path.join(save, "images", save_name))

    if os.path.exists(gt_path):
        shutil.copy(gt_path, os.path.join(save, "gt", save_name))
    else:
        H, W, _ = cv2.imread(p).shape
        cv2.imwrite(os.path.join(save, "gt", save_name), np.zeros((H, W)))
