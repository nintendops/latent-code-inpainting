
# Pluralistic Image Inpainting with Latent Codes

*This readme is still a work in progress, some links may not work*

[paper](https://arxiv.org/html/2403.18186v1) | [arXiv](https://arxiv.org/abs/2403.18186)

This repository contains the code (in PyTorch) for ''Don't Look into the Dark: Latent Codes for Pluralistic Image Inpainting'' (CVPR'2024) by Haiwei Chen and [Yajie Zhao](https://www.yajie-zhao.com/).



## Contents

1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Contact](#contact)

## Getting Started
The inpainting method in this repository utilizes priors learnt from discrete latent codes to diversely complete a masked image. The method works in both free-form and large-hole mask settings: 

![](https://github.com/nintendops/latent-code-inpainting/meida/github_inpainting.gif)

## Requirements

The code has been tested on Python3.11, PyTorch 2.1.0 and CUDA (12.1). The additional dependencies can be installed with 
```
python install -r environment.txt
```

## Getting Started

Our models are built upon training data from both [Places365-Standard](http://places2.csail.mit.edu/download-private.html) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans).

As the first step, please download the respective pretrained models ([Places]() | [CelebA-HQ]()) and places the checkpoint files under the ckpts folder in the root directory.
 

**Quick Test**

We provide a [demo notebook](https://github.com/nintendops/latent-code-inpainting/blob/main/eval.ipynb) for quickly testing the inpainting models. Please follow instructions in the notebook to set up inference with your desired configurations.

**Training**

If you are interested in training our models on custom data, please ...

```
# modelnet classification
CUDA_VISIBLE_DEVICES=0 python run_modelnet.py experiment -d PATH_TO_MODELNET40
# modelnet shape alignment
CUDA_VISIBLE_DEVICES=0 python run_modelnet_rotation.py experiment -d PATH_TO_MODELNET40
# 3DMatch shape registration
CUDA_VISIBLE_DEVICES=0 python run_3dmatch.py experiment -d PATH_TO_3DMATCH
```


## Contact
Haiwei Chen: chw9308@hotmail.com
Any discussions or concerns are welcomed!

**Citation**
If you find our project useful in your research, please consider citing:

```
@article{chen2024don,
  title={Don't Look into the Dark: Latent Codes for Pluralistic Image Inpainting},
  author={Chen, Haiwei and Zhao, Yajie},
  journal={arXiv preprint arXiv:2403.18186},
  year={2024}
}
```
## License and Acknowledgement
The code and models in this repo are for research purposes only. Our code is bulit upon [VQGAN](https://github.com/CompVis/taming-transformers).
