import glob
import os
import shutil

import cv2
import numpy as np
import tqdm

root = "data/industral/BTAD/VTADL/btad/BTech_Dataset_transformed/"
save = "converted_data/btad"
os.makedirs(os.path.join(save, "images"), exist_ok=True)
os.makedirs(os.path.join(save, "gt"), exist_ok=True)

for p in tqdm.tqdm(glob.glob(f"{root}/*/test/*/*")):
    gt_path = p.replace("test", "ground_truth")
    if "/01/" in gt_path:
        gt_path = gt_path.replace(".bmp", ".png")
    p_list = p.split(os.path.sep)
    save_name = p_list[-4] + "_" + p_list[-2] + "_" + p_list[-1][:-4] + ".png"
    shutil.copy(p, os.path.join(save, "images", save_name))
    if "ok" in gt_path:
        H, W, _ = cv2.imread(p).shape
        cv2.imwrite(os.path.join(save, "gt", save_name), np.zeros((H, W)))
    else:
        shutil.copy(gt_path, os.path.join(save, "gt", save_name))
