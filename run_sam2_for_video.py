import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from utils.clicker import prompt_sam2_with_point
from utils.py_utils import generate_whole_bbox, load_mask, perturb_bounding_boxes, random_erose_or_dilate


def get_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Run SAM 2 for evaluating video segmentation performance and robustness.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a yaml config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to either a single input image or folder of images.")
    parser.add_argument("--output", type=str, required=True, help="Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single json with COCO-style masks.")
    parser.add_argument("--prompt_type", type=str, required=True, choices=["bbox", "point", "mask"])
    parser.add_argument("--perturbation", action='store_true', help="Apply perturbation to the prompt of the current frame.")
    parser.add_argument("--num_prompts", type=int, required=True, choices=[1, 3, 5])
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    # fmt: on

    args = parser.parse_args()
    with open(args.config, mode="r", encoding="utf-8") as f:
        args.config = yaml.safe_load(f)
    dataset_info = args.config["dataset"][args.dataset]

    args.image_seq_roots = []
    args.gt_seq_roots = []
    for image_seq_root in Path(dataset_info["image"]["root"]).iterdir():
        gt_seq_root = Path(dataset_info["mask"]["root"]).joinpath(image_seq_root.stem)
        if gt_seq_root.is_dir():
            args.image_seq_roots.append(image_seq_root)
            args.gt_seq_roots.append(gt_seq_root)

    args.image_subdir = dataset_info["image"]["subdir"]
    args.image_suffix = dataset_info["image"]["suffix"]
    args.gt_subdir = dataset_info["mask"]["subdir"]
    args.gt_suffix = dataset_info["mask"]["suffix"]

    args.proj_name = f"SAM2-wPrompt{args.prompt_type}{args.num_prompts}-for-Video"
    if args.perturbation:
        args.proj_name += "Robustness"
    else:
        args.proj_name += "Performance"

    args.output = Path(args.output)
    args.proj_path = args.output.joinpath(args.proj_name)
    args.proj_path.mkdir(parents=True, exist_ok=True)
    return args


def load_sam_model(args):
    predictor = build_sam2_video_predictor(
        "sam2_hiera_l.yaml", args.config["checkpoint"]["sam2-l"], device=args.device
    )
    return predictor


def main() -> None:
    args = get_args()

    print("Loading Model...")
    predictor = load_sam_model(args)

    print(f"Processing Dataset {args.dataset} ({len(args.image_seq_roots)}, {len(args.gt_seq_roots)})...")
    num_seqs = len(args.image_seq_roots)
    for seq_idx, (image_seq_root, gt_seq_root) in enumerate(zip(args.image_seq_roots, args.gt_seq_roots)):
        seq_name = image_seq_root.stem

        image_subdir = Path(image_seq_root).joinpath(args.image_subdir)
        gt_subdir = Path(gt_seq_root).joinpath(args.gt_subdir)

        image_paths = []
        gt_paths = []
        for image_path in tqdm(sorted(image_subdir.iterdir(), key=lambda p: p.stem)):
            gt_path = gt_subdir.joinpath(image_path.with_suffix(args.gt_suffix).name)
            if not (
                gt_path.exists()
                and image_path.as_posix().endswith(args.image_suffix)
                and gt_path.as_posix().endswith(args.gt_suffix)
            ):
                raise ValueError(f"{image_path.as_posix()} and {gt_path.as_posix()} must exist at the same time.")
            image_paths.append(image_path)
            gt_paths.append(gt_path)
        num_frames = len(gt_paths)
        assert num_frames > 0, (image_subdir.as_posix(), gt_subdir.as_posix())

        print(f"[{seq_idx}/{num_seqs}] Processing Video {seq_name} ({len(image_paths)}, {len(gt_paths)})...")
        inference_state = predictor.init_state(video_path=image_subdir.as_posix())
        predictor.reset_state(inference_state)

        # set the prompts
        prompt_indices = [i * (num_frames // args.num_prompts) for i in range(args.num_prompts)]
        print(f"prompt_indices: {prompt_indices}")
        for prompt_idx in prompt_indices:
            bin_gt, uint8_gt = load_mask(path=gt_paths[prompt_idx], return_uint8=True)

            if args.prompt_type == "bbox":
                bboxes = [generate_whole_bbox(bin_gt)]

                if args.perturbation:
                    bboxes = perturb_bounding_boxes(bboxes, bin_gt.shape[0], bin_gt.shape[1])

                for obj_id, bbox in enumerate(bboxes):
                    predictor.add_new_points_or_box(
                        inference_state=inference_state, frame_idx=prompt_idx, obj_id=obj_id, box=bbox
                    )

            elif args.prompt_type == "mask":
                if args.perturbation:
                    bin_gt = random_erose_or_dilate(bin_gt).astype(bool)
                predictor.add_new_mask(inference_state=inference_state, frame_idx=prompt_idx, obj_id=0, mask=bin_gt)

            elif args.prompt_type == "point":
                prompt_sam2_with_point(
                    gt_mask=uint8_gt,
                    predictor=predictor,
                    max_iou_thr=0.9,
                    max_clicks=6,
                    inference_state=inference_state,
                    frame_idx=prompt_idx,
                    perturbation=args.perturbation,
                )

        # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_obj_logits in predictor.propagate_in_video(inference_state):
            final_mask = np.zeros_like(uint8_gt, dtype=bool)
            for _id, _logits in zip(out_obj_ids, out_obj_logits):
                _mask = (_logits > 0.0).squeeze().cpu().numpy()
                final_mask = np.logical_or(final_mask, _mask)

            final_mask = (final_mask.astype(np.uint8)) * 255
            save_path = Path(args.proj_path).joinpath(
                args.dataset, seq_name, args.gt_subdir, gt_paths[out_frame_idx].name
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path.as_posix(), final_mask)


if __name__ == "__main__":
    main()
