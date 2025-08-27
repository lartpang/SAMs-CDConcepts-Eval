# Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes

<div align="center">
   <a href='https://arxiv.org/abs/2412.01240'>
      <img src='https://img.shields.io/badge/ArXiv-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
   </a>
  <img src="https://img.shields.io/github/last-commit/lartpang/SAMs-CDConcepts-Eval">
</div>

As large-scale foundation models trained on billions of image–mask pairs covering a vast diversity of scenes, objects, and contexts, SAM and its upgraded version, SAM 2, have significantly influenced multiple fields within computer vision. Leveraging such unprecedented data diversity, they exhibit strong open-world segmentation capabilities, with SAM 2 further enhancing these capabilities to support high-quality video segmentation. 
While SAMs (SAM and SAM 2) have demonstrated excellent performance in segmenting context-independent concepts like people, cars, and roads, they overlook more challenging context-dependent (CD) concepts, such as visual saliency, camouflage, industrial defects, and medical lesions. CD concepts rely heavily on global and local contextual information, making them susceptible to shifts in different contexts, which requires strong discriminative capabilities from the model. 
The lack of comprehensive evaluation of SAMs limits understanding of their performance boundaries, which may hinder the design of future models. In this paper, we conduct a thorough evaluation of SAMs on 11 CD concepts across 2D and 3D images and videos in various visual modalities within natural, medical, and industrial scenes. We develop a unified evaluation framework for SAM and SAM 2 that supports manual, automatic, and intermediate self-prompting, aided by our specific prompt generation and interaction strategies. We further explore the potential of SAM 2 for in-context learning and introduce prompt robustness testing to simulate real-world imperfect prompts. Finally, we analyze the benefits and limitations of SAMs in understanding CD concepts and discuss their future development in segmentation tasks. This work aims to provide valuable insights to guide future research in both context-independent and context-dependent concepts segmentation, potentially informing the development of the next version — SAM 3.

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
   1. `git clone https://github.com/facebookresearch/sam2.git segment-anything2` (Use a separate folder for `sam2` code.)
   2. `cd segment-anything2`
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

## Awesome-Unified-Context-dependent-Concept-Segmentation

- <https://github.com/Xiaoqi-Zhao-DLUT/Awesome-Unified-Context-dependent-Concept-Segmentation>

## Contributors

- [@lartpang](https://github.com/lartpang)
- [@Xiaoqi-Zhao-DLUT](https://github.com/Xiaoqi-Zhao-DLUT)
- [@DUT-CSJ](https://github.com/DUT-CSJ)
- [@zhaoyuan1209](https://github.com/zhaoyuan1209)

## Citation

```bibtex
@misc{Eva_SAMs,
      title={Inspiring the Next Generation of Segment Anything Models: Comprehensively Evaluate SAM and SAM 2 with Diverse Prompts Towards Context-Dependent Concepts under Different Scenes}, 
      author={Xiaoqi Zhao and Youwei Pang and Shijie Chang and Yuan Zhao and Lihe Zhang and  Chenyang Yu and Hanqi Liu and Jiaming Zuo and Jinsong Ouyang and Weisi Lin and Georges El Fakhri and Huchuan Lu and Xiaofeng Liu},
      year={2025},
      eprint={2412.01240},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01240}, 
}
```
