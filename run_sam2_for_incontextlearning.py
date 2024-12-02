import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from skimage.transform import resize

from utils.py_utils import load_mask


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description="Run ICL-based SAM 2 for evaluating image segmentation performance")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to a yaml config file.")
    parser.add_argument("--prompt_dataset", type=str, required=True, help="Path to the folder of images.")
    parser.add_argument("--num_prompts", type=int, default=20, help="Number of prompts to use.")
    parser.add_argument("--dataset", type=str, default="DUTS-TE", help="Path to the folder of images.")
    parser.add_argument("--output", type=str, default="prediction/SAMl/SOD/DUTS", help=("Path to the directory where masks will be output. Output will be either a folder. "))
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    # fmt: on

    args = parser.parse_args()
    with open(args.config, mode="r", encoding="utf-8") as f:
        args.config = yaml.safe_load(f)

    args.model = "sam2-l"

    args.prompt_image_dir = Path(args.config["dataset"][args.prompt_dataset]["image"]["root"])
    assert args.prompt_image_dir.is_dir(), "image_dir must be a folder."
    args.prompt_image_suffix = args.config["dataset"][args.prompt_dataset]["image"]["suffix"]

    args.prompt_gt_dir = Path(args.config["dataset"][args.prompt_dataset]["mask"]["root"])
    assert args.prompt_gt_dir.is_dir(), "gt_dir must be a folder."
    args.prompt_gt_suffix = args.config["dataset"][args.prompt_dataset]["mask"]["suffix"]

    args.image_dir = Path(args.config["dataset"][args.dataset]["image"]["root"])
    assert args.image_dir.is_dir(), "image_dir must be a folder."
    args.image_suffix = args.config["dataset"][args.dataset]["image"]["suffix"]

    args.gt_dir = Path(args.config["dataset"][args.dataset]["mask"]["root"])
    assert args.gt_dir.is_dir(), "gt_dir must be a folder."
    args.gt_suffix = args.config["dataset"][args.dataset]["mask"]["suffix"]

    args.proj_name = f"{args.model.upper()}-wICL-for-ImagePerformance"

    args.output = Path(args.output)
    args.proj_path = args.output.joinpath(args.proj_name)
    args.proj_path.mkdir(parents=True, exist_ok=True)
    return args


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
):
    hydra_overrides = [
        "++model._target_=utils.sam2_video_predictor.ICLSAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def load_sam_model(args):
    predictor = build_sam2_video_predictor(
        "sam2_hiera_l.yaml", args.config["checkpoint"]["sam2-l"], device=args.device
    )
    return predictor


def main() -> None:
    args = get_args()

    print("Loading model...")
    predictor = load_sam_model(args)

    print(f"Loading prompt data ({args.prompt_dataset})")
    prompt_image_paths = []
    for prompt_image_path in args.prompt_image_dir.iterdir():
        prompt_gt_path: Path = args.prompt_gt_dir / prompt_image_path.with_suffix(args.gt_suffix).name
        if prompt_gt_path.exists():
            prompt_image_paths.append(prompt_image_path)
    prompt_image_paths = sorted(prompt_image_paths, key=lambda x: x.stem)[: args.num_prompts]
    prompt_gt_paths = []
    for prompt_image_path in prompt_image_paths:
        prompt_gt_path: Path = args.prompt_gt_dir / prompt_image_path.with_suffix(args.gt_suffix).name
        prompt_gt_paths.append(prompt_gt_path)
    assert len(prompt_image_paths) == len(prompt_image_paths) == args.num_prompts

    image_paths = []
    gt_paths = []
    for image_path in args.image_dir.iterdir():
        gt_path: Path = args.gt_dir / image_path.with_suffix(args.gt_suffix).name
        if gt_path.exists():
            image_paths.append(image_path)
            gt_paths.append(gt_path)

    print(f"Start processing dataset {args.dataset} ({len(image_paths)}, {len(gt_paths)})...")
    group_prompts = predictor.init_group_prompts(prompt_image_paths=prompt_image_paths)
    for image_path, gt_path in zip(image_paths, gt_paths):
        inference_state = predictor.init_state(image_path=image_path.as_posix(), group_prompts=group_prompts)
        predictor.reset_state(inference_state)

        # import the gt masks for those prompt images
        for i, prompt_gt_path in enumerate(prompt_gt_paths):
            bin_gt = load_mask(prompt_gt_path)
            predictor.add_new_mask(inference_state=inference_state, frame_idx=i, obj_id=0, mask=bin_gt)

        # video_segments contains the per-frame segmentation results
        video_segments = {}  # out_frame_idx starts from 0
        for out_frame_idx, out_obj_ids, out_obj_logits in predictor.propagate_in_video(inference_state):
            if out_frame_idx >= args.num_prompts:
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_obj_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
                }
        frame_segments = video_segments[out_frame_idx]  # only use the final frame

        bin_gt, uint8_gt = load_mask(gt_path, return_uint8=True)
        final_mask = np.zeros_like(bin_gt, dtype=bool)
        for out_obj_id, out_obj_mask in frame_segments.items():
            resized_out_mask = resize(
                out_obj_mask[0], final_mask.shape, order=0, preserve_range=True, anti_aliasing=False
            )
            final_mask = np.logical_or(final_mask, resized_out_mask)

        final_mask = (final_mask.astype(np.uint8)) * 255
        save_path = Path(args.proj_path).joinpath(args.dataset, gt_path.name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path.as_posix(), final_mask)


if __name__ == "__main__":
    main()
