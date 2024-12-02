import glob
import os
import random

import cv2
import nibabel as nib
import numpy as np
import tqdm


def nii2pngs(nii_path):
    nii_data = nib.load(nii_path)
    img_data = nii_data.get_fdata()

    res = []
    for i in range(img_data.shape[2]):
        slice_data = img_data[:, :, i]
        # minmax
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))
        slice_data = (slice_data * 255).astype(np.uint8)
        res.append(slice_data)
    return res


root = "data/Brats2020/MICCAI_BraTS2020_TrainingData"
save_root = "converted_data/brats/brats_2020"

all_segs = glob.glob(f"{root}/*/*_seg.nii")
all_segs = random.sample(all_segs, k=37)
for seg_cls in ["WT", "ET", "TC"]:
    for moda in ["flair", "t1", "t1ce", "t2"]:
        save = save_root + "_" + moda + "_" + seg_cls

        paths = []
        for gt_path in tqdm.tqdm(all_segs):
            p = gt_path.replace("seg", moda)
            p_list = p.split(os.path.sep)
            save_name = p_list[-1]

            gts = nii2pngs(gt_path)
            gts = np.stack(gts)
            if seg_cls == "WT":
                gts = (gts > 0) * 255
            elif seg_cls == "ET":
                gts = (gts == 255) * 255
            elif seg_cls == "TC":
                gts[gts == 63] = 255
                gts = (gts == 255) * 255

            images = nii2pngs(p)
            images = np.stack(images)
            index = np.argmax(gts.sum((1, 2)), axis=0)
            for i, (gt, img) in enumerate(zip(gts[: index + 1][::-1], images[: index + 1][::-1])):
                if i == 0:
                    paths.append(os.path.join(save, "gt", save_name + "_former"))
                os.makedirs(os.path.join(save, "gt", save_name + "_former"), exist_ok=True)
                os.makedirs(os.path.join(save, "videos", save_name + "_former"), exist_ok=True)
                cv2.imwrite(os.path.join(save, "gt", save_name + "_former", f"{i}.jpg".zfill(10)), gt)
                cv2.imwrite(os.path.join(save, "videos", save_name + "_former", f"{i}.jpg".zfill(10)), img)

            for i, (gt, img) in enumerate(zip(gts[index:], images[index:])):
                os.makedirs(os.path.join(save, "gt", save_name + "_latter"), exist_ok=True)
                os.makedirs(os.path.join(save, "videos", save_name + "_latter"), exist_ok=True)
                cv2.imwrite(os.path.join(save, "gt", save_name + "_latter", f"{i}.jpg".zfill(10)), gt)
                cv2.imwrite(os.path.join(save, "videos", save_name + "_latter", f"{i}.jpg".zfill(10)), img)

        print(len(os.listdir(os.path.join(save, "videos"))))
