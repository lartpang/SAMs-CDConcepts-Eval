import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from tqdm import tqdm

from utils.clicker import prompt_sams_with_point
from utils.py_utils import find_connect_area, load_image, load_mask, perturb_bounding_boxes


def get_args() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser(description="Run SAM and SAM 2 for evaluating image segmentation performance and robustness.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a yaml config file.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to either a single input image or folder of images.")
    parser.add_argument("--output", type=str, default="output", help="Path to the directory where masks will be output.")
    parser.add_argument("--model", type=str, required=True, choices=["sam-l", "sam2-l"])
    parser.add_argument("--prompt_type", type=str, required=True, choices=["auto", "bbox", "point"])
    parser.add_argument("--perturbation", action='store_true', help="Apply perturbation to the prompt of the current frame.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    amg_settings = parser.add_argument_group("AMG Settings")
    amg_settings.add_argument("--points-per-side", type=int, help="Generate masks by sampling a grid over the image with this many points to a side.")
    amg_settings.add_argument("--points-per-batch", type=int, help="How many input points to process simultaneously in one batch.")
    amg_settings.add_argument("--pred-iou-thresh", type=float, help="Exclude masks with a predicted score from the model that is lower than this threshold.")
    amg_settings.add_argument("--stability-score-thresh", type=float, help="Exclude masks with a stability score lower than this threshold.")
    amg_settings.add_argument("--stability-score-offset", type=float, help="Larger values perturb the mask more when measuring stability score.")
    amg_settings.add_argument("--box-nms-thresh", type=float, help="The overlap threshold for excluding a duplicate mask.")
    amg_settings.add_argument("--crop-n-layers", type=int, help= "If >0, mask generation is run on smaller crops of the image to generate more masks. The value sets how many different scales to crop at.")
    amg_settings.add_argument("--crop-nms-thresh", type=float, default=None, help="The overlap threshold for excluding duplicate masks across different crops.")
    amg_settings.add_argument("--crop-overlap-ratio", type=int, default=None, help="Larger numbers mean image crops will overlap more.")
    amg_settings.add_argument("--crop-n-points-downscale-factor", type=int, default=None, help="The number of points-per-side in each layer of crop is reduced by this factor.")
    amg_settings.add_argument("--min-mask-region-area", type=int, default=None, help="Disconnected mask regions or holes with area smaller than this value in pixels are removed by postprocessing.")
    # fmt: on

    args = parser.parse_args()
    with open(args.config, mode="r", encoding="utf-8") as f:
        args.config = yaml.safe_load(f)
    dataset_info = args.config["dataset"][args.dataset]

    args.dataset_info = dataset_info
    args.image_root = Path(dataset_info["image"]["root"])
    args.image_suffix = dataset_info["image"]["suffix"]
    args.gt_root = Path(dataset_info["mask"]["root"])
    args.gt_suffix = dataset_info["mask"]["suffix"]

    args.proj_name = f"{args.model.upper()}-wPrompt{args.prompt_type}-for-Image"
    if args.perturbation:
        args.proj_name += "Robustness"
    else:
        args.proj_name += "Performance"

    args.output = Path(args.output)
    args.proj_path = args.output.joinpath(args.proj_name)
    args.proj_path.mkdir(parents=True, exist_ok=True)
    return args


def load_sam_model(args):
    if args.model == "sam-l":
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

    elif args.model == "sam2-l":
        sam2 = build_sam2("sam2_hiera_l.yaml", args.config["checkpoint"]["sam2-l"], apply_postprocessing=False)
        sam2 = sam2.to(device=args.device)

        if args.prompt_type == "auto":
            predictor = SAM2AutomaticMaskGenerator(model=sam2)
        elif args.prompt_type == "bbox" or args.prompt_type == "point":
            predictor = SAM2ImagePredictor(sam2)
        else:
            raise ValueError(f"Unsupported prompt type: {args.prompt_type}")

    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    return predictor


def main() -> None:
    args = get_args()

    print("Loading model...")
    predictor = load_sam_model(args)

    print(f"Loading dataset ({args.dataset}) info: {args.dataset_info}")
    image_paths = []
    gt_paths = []
    for image_path in args.image_root.iterdir():
        gt_path = args.gt_root / image_path.with_suffix(args.gt_suffix).name
        if gt_path.is_file():
            image_paths.append(image_path)
            gt_paths.append(gt_path)

    print(f"Start processing dataset {args.dataset} ({len(image_paths)}, {len(gt_paths)})...")
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths), ncols=78):
        image = load_image(image_path)
        bin_gt, uint8_gt = load_mask(gt_path, return_uint8=True)

        if args.prompt_type == "auto":
            assert args.perturbation is False

            final_mask = np.zeros_like(bin_gt, dtype=bool)
            for mask in predictor.generate(image):
                segmentation = mask["segmentation"]
                intersection = np.logical_and(segmentation, bin_gt)
                if np.sum(intersection) / np.sum(segmentation) > 0.9:
                    final_mask = np.logical_or(final_mask, segmentation)

        elif args.prompt_type == "bbox":
            predictor.set_image(image)

            bboxes = find_connect_area(bin_gt)
            if args.perturbation:
                bboxes = perturb_bounding_boxes(bboxes, bin_gt.shape[0], bin_gt.shape[1])

            masks = []
            for bbox in bboxes:
                mask = predictor.predict(box=bbox, multimask_output=False)[0]
                masks.append(mask[0])

            final_mask = np.zeros_like(bin_gt, dtype=bool)
            for mask in masks:
                final_mask = np.logical_or(final_mask, mask)

        elif args.prompt_type == "point":
            final_mask = prompt_sams_with_point(
                image=image,
                gt_mask=uint8_gt,
                predictor=predictor,
                max_iou_thr=0.9,
                max_clicks=6,
                perturbation=args.perturbation,
            )

        final_mask = (final_mask.astype(np.uint8)) * 255
        save_path = Path(args.proj_path).joinpath(args.dataset, gt_path.name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path.as_posix(), final_mask)


if __name__ == "__main__":
    main()
