# Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes

## Usage

### Prepare Datasets

See our ArXiv version for dataset details and set their paths in the config file.

> [!note]
> - Some datasets need to be preprocessed before use by the scripts in the folder [`preprocess`](./preprocess/):
> - Images in some datasets like CAD for Video COD, may not have corresponding annotations, and these images will not be used for model prediction and performance evaluation. So please clean them up in advance before the script is used.

### Prepare SAM and SAM 2

1. Install SAM:
   1. `git clone https://github.com/facebookresearch/segment-anything.git`
   2. `cd segment-anything`
   3. `pip install -e .`
2. Install SAM 2:
   1. `git clone https://github.com/facebookresearch/sam2.git`
   2. `cd sam2`
   3. `pip install -e .`
3. Download SAM and SAM 2 checkpoints and assign their paths to the items `sam-l` and `sam2-l` of the config file:
   1. `vit_l` checkpoint from <https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints>.
      1. url: <https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth>
   2. `hiera_large` checkpoint from <https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-2-checkpoints>
      1. url: <https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt>

### Generate Predictions

Run the corresponding commands (see [./run.sh](./run.sh)) to generate predictions for each task.

## Evaluation Tools

- <https://github.com/Xiaoqi-Zhao-DLUT/PySegMetric_EvalToolkit>
- <https://github.com/zhaoyuan1209/PyADMetric_EvalToolkit>
