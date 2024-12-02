import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm

from utils.clicker import prompt_sams_with_point
from utils.py_utils import find_connect_area, load_image, load_mask, perturb_bounding_boxes


def get_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Run SAM for evaluating video segmentation performance and robustness.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a yaml config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to either a single input image or folder of images.")
    parser.add_argument("--output", type=str, required=True, help="Path to the directory where masks will be output. Output will be either a folder of PNGs per image or a single json with COCO-style masks.")
    parser.add_argument("--prompt_type", type=str, required=True, choices=["auto", "bbox", "point"])
    parser.add_argument("--perturbation", action='store_true', help="Apply perturbation to the prompt of the current frame.")
    parser.add_argument("--propagation", action='store_true', help="Propagate prediction to generate the prompt of the next frame.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    amg_settings = parser.add_argument_group("AMG Settings")
    amg_settings.add_argument("--points-per-side", type=int, default=None, help="Generate masks by sampling a grid over the image with this many points to a side.")
    amg_settings.add_argument("--points-per-batch", type=int, default=None, help="How many input points to process simultaneously in one batch.")
    amg_settings.add_argument("--pred-iou-thresh", type=float, default=None, help="Exclude masks with a predicted score from the model that is lower than this threshold.")
    amg_settings.add_argument("--stability-score-thresh", type=float, default=None, help="Exclude masks with a stability score lower than this threshold.")
    amg_settings.add_argument("--stability-score-offset", type=float, default=None, help="Larger values perturb the mask more when measuring stability score.")
    amg_settings.add_argument("--box-nms-thresh", type=float, default=None, help="The overlap threshold for excluding a duplicate mask.")
    amg_settings.add_argument("--crop-n-layers", type=int, default=None, help="If >0, mask generation is run on smaller crops of the image to generate more masks. The value sets how many different scales to crop at.")
    amg_settings.add_argument("--crop-nms-thresh", type=float, default=None, help="The overlap threshold for excluding duplicate masks across different crops.")
    amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None, help="Larger numbers mean image crops will overlap more.")
    amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None, help="The number of points-per-side in each layer of crop is reduced by this factor.")
    amg_settings.add_argument("--min-mask-region-area", type=int, default=None, help="Disconnected mask regions or holes with area smaller than this value in pixels are removed by postprocessing.")
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

    args.proj_name = "SAM-w"
    if args.propagation:
        args.proj_name += "Propagated"
    args.proj_name += f"Prompt{args.prompt_type}-for-Video"
    if args.perturbation:
        args.proj_name += "Robustness"
    else:
        args.proj_name += "Performance"

    args.output = Path(args.output)
    args.proj_path = args.output.joinpath(args.proj_name)
    args.proj_path.mkdir(parents=True, exist_ok=True)
    return args


def load_sam_model(args):
    sam = sam_model_registry["vit_l"](checkpoint=args.config["checkpoint"]["sam-l"])
    sam = sam.to(device=args.device)

    if args.prompt_type == "auto":
        amg_kwargs = {
            "points_per_side": args.points_per_side,
            "points_per_batch": args.points_per_batch,
            "pred_iou_thresh": args.pred_iou_thresh,
            "stability_score_thresh": args.stability_score_thresh,
            "stability_score_offset": args.stability_score_offset,
            "box_nms_thresh": args.box_nms_thresh,
            "crop_n_layers": args.crop_n_layers,
            "crop_nms_thresh": args.crop_nms_thresh,
            "crop_overlap_ratio": args.crop_overlap_ratio,
            "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
            "min_mask_region_area": args.min_mask_region_area,
        }
        amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}

        predictor = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", **amg_kwargs)
    elif args.prompt_type == "bbox" or args.prompt_type == "point":
        predictor = SamPredictor(sam)
    else:
        raise ValueError(f"Unsupported prompt type: {args.prompt_type}")
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
        propagated_prompt = None
        for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), ncols=78):
            image = load_image(image_path)
            bin_gt, uint8_gt = load_mask(gt_path, return_uint8=True)

            if args.prompt_type == "auto":
                assert args.propagation is False and args.perturbation is False

                final_mask = np.zeros_like(bin_gt, dtype=bool)
                for mask in predictor.generate(image):
                    segmentation = mask["segmentation"]  # numpy.ndarray[bool]
                    intersection = np.logical_and(segmentation, bin_gt)
                    if np.sum(intersection) / np.sum(segmentation) > 0.9:
                        final_mask = np.logical_or(final_mask, segmentation)

            elif args.prompt_type == "bbox":
                predictor.set_image(image)

                if not args.propagation or propagated_prompt is None:
                    bboxes = find_connect_area(bin_gt)
                else:
                    bboxes = propagated_prompt

                if args.perturbation:
                    bboxes = perturb_bounding_boxes(bboxes, bin_gt.shape[0], bin_gt.shape[1])

                masks = []
                for bbox in bboxes:
                    mask = predictor.predict(box=bbox, multimask_output=False)[0]
                    masks.append(mask[0])

                final_mask = np.zeros_like(bin_gt, dtype=bool)
                for mask in masks:
                    final_mask = np.logical_or(final_mask, mask)

                if args.propagation:
                    propagated_prompt = find_connect_area(final_mask)

            elif args.prompt_type == "point":
                final_mask = prompt_sams_with_point(
                    image=image,
                    gt_mask=uint8_gt,
                    predictor=predictor,
                    max_iou_thr=0.9,
                    max_clicks=6,
                    perturbation=args.perturbation,
                    prompt_mask=propagated_prompt,
                )
                if args.propagation:
                    propagated_prompt = final_mask.astype(np.uint8)

            final_mask = (final_mask.astype(np.uint8)) * 255
            save_path = Path(args.proj_path).joinpath(args.dataset, seq_name, args.gt_subdir, gt_path.name)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path.as_posix(), final_mask)


if __name__ == "__main__":
    main()
