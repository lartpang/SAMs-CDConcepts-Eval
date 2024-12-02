#!/usr/bin/env bash
set -e          # The script stops running whenever an error occurs.
set -u          # If it encounters a non-existent variable it reports an error and stops execution.
set -x          # When running the result of a command, output the corresponding command.
set -o pipefail # Ensure that whenever a subcommand fails, the entire pipeline command fails.


# Corresponding to Tabs. 2-8 of the paper:
# - SAM image segmentation performance with basic prompts
python run_sams_for_image.py --config config.yaml --model sam-l --dataset ECSSD --output output --prompt_type bbox
python run_sams_for_image.py --config config.yaml --model sam-l --dataset ECSSD --output output --prompt_type point
python run_sams_for_image.py --config config.yaml --model sam-l --dataset ECSSD --output output --prompt_type auto
# - SAM 2 image segmentation performance with basic prompts
python run_sams_for_image.py --config config.yaml --model sam2-l --dataset ECSSD --output output --prompt_type bbox
python run_sams_for_image.py --config config.yaml --model sam2-l --dataset ECSSD --output output --prompt_type point
python run_sams_for_image.py --config config.yaml --model sam2-l --dataset ECSSD --output output --prompt_type auto

# Corresponding to Tab. 9 of the paper:
# - SAM 2 image segmentation performance with the in-context learning mode
python run_sam2_for_incontextlearning.py --config config.yaml --prompt_dataset DUTS-TR --num_prompts 20 --dataset DUTS-TE --output output

# Corresponding to Tabs. 10-13 of the paper:
# - SAM video segmentation performance with propagated prompts
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox --propagation
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point --propagation
# - SAM video segmentation performance with basic prompts
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type auto
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point
# - SAM 2 video segmentation performance with basic prompts
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox --num_prompts 5
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point --num_prompts 5
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type mask --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type mask --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type mask --num_prompts 5

# Corresponding to Tab. 14 of the paper:
# - SAM 2 3D LOS segmentation performance with basic prompts
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type bbox --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type bbox --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type bbox --num_prompts 5
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type point --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type point --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type point --num_prompts 5
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type mask --num_prompts 1
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type mask --num_prompts 3
python run_sam2_for_video.py --config config.yaml --dataset BraTS2020-Flair --output output --prompt_type mask --num_prompts 5

# Corresponding to Tab. 15 of the paper:
# - SAM image segmentation robustness with perturbed prompts
python run_sams_for_image.py --config config.yaml --model sam-l --dataset DUTS-TE --output output --prompt_type bbox --perturbation
python run_sams_for_image.py --config config.yaml --model sam-l --dataset DUTS-TE --output output --prompt_type point --perturbation
# - SAM 2 image segmentation robustness with perturbed prompts
python run_sams_for_image.py --config config.yaml --model sam2-l --dataset DUTS-TE --output output --prompt_type bbox --perturbation
python run_sams_for_image.py --config config.yaml --model sam2-l --dataset DUTS-TE --output output --prompt_type point --perturbation
# - SAM image segmentation robustness with perturbed prompts
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type bbox --propagation
python run_sam1_for_video.py --config config.yaml --dataset DAVIS16-Val --output output --prompt_type point --propagation
# - SAM 2 image segmentation robustness with perturbed prompts
python run_sam2_for_video.py --config config.yaml --num_prompts 1 --dataset DAVIS16-Val --output output --prompt_type bbox --perturbation
python run_sam2_for_video.py --config config.yaml --num_prompts 1 --dataset DAVIS16-Val --output output --prompt_type point --perturbation
python run_sam2_for_video.py --config config.yaml --num_prompts 1 --dataset DAVIS16-Val --output output --prompt_type mask --perturbation
