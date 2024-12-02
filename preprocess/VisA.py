import os
import shutil

import cv2
import numpy as np
import pandas as pd
import tqdm

root = "data/visa"
save = "converted_data/visa"
os.makedirs(os.path.join(save, "images"), exist_ok=True)
os.makedirs(os.path.join(save, "gt"), exist_ok=True)

visa_list = pd.read_csv(f"{root}/split_csv/1cls.csv")
visa_list = visa_list[visa_list["split"] == "test"]["image"].to_list()
visa_list = [os.path.join(root, p) for p in visa_list]
for p in tqdm.tqdm(visa_list):
    gt_path = p.replace("Images", "Masks").replace(".JPG", ".png")
    p_list = p.split(os.path.sep)
    save_name = p_list[-5] + "_" + p_list[-2] + "_" + p_list[-1].replace(".JPG", ".png")
    shutil.copy(p, os.path.join(save, "images", save_name))
    if os.path.exists(gt_path):
        shutil.copy(gt_path, os.path.join(save, "gt", save_name))
    else:
        H, W, _ = cv2.imread(p).shape
        cv2.imwrite(os.path.join(save, "gt", save_name), np.zeros((H, W)))
