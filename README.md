

# Pluralistic Image Inpainting with Latent Codes


[paper]([https://arxiv.org/html/2403.18186v1](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Dont_Look_into_the_Dark_Latent_Codes_for_Pluralistic_Image_CVPR_2024_paper.html) | [arXiv](https://arxiv.org/abs/2403.18186)

This repository contains the code (in PyTorch) for ''Don't Look into the Dark: Latent Codes for Pluralistic Image Inpainting'' (CVPR'2024) by Haiwei Chen and [Yajie Zhao](https://www.yajie-zhao.com/).



## Contents

1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Experiments](#experiments)
4. [Contact](#contact)

## Getting Started
The inpainting method in this repository utilizes priors learnt from discrete latent codes to diversely complete a masked image. The method works in both free-form and large-hole mask settings: 

![](https://github.com/nintendops/latent-code-inpainting/blob/main/media/main.gif?raw=true)

## Requirements

The code has been tested on Python3.11, PyTorch 2.1.0 and CUDA (12.1). The additional dependencies can be installed with 
```
pip install -r environment.txt
```

## Getting Started

Our models are built upon training data from both [Places365-Standard](http://places2.csail.mit.edu/download-private.html) and [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans).

As the first step, please download the respective pretrained models ([Places](https://drive.google.com/drive/folders/1ZchB85kuUjLpxcz-WSgPDbkfeFcPRjZL?usp=sharing) | [CelebA-HQ](https://drive.google.com/drive/folders/1-o9KefXQb7R8qE70luYU58u-ksXOgmBh?usp=sharing)) and places the checkpoint files under the ```ckpts/``` folder in the root directory.
 

**Quick Test**

We provide a [demo notebook](https://github.com/nintendops/latent-code-inpainting/blob/main/eval.ipynb) at ```eval.ipynb``` for quickly testing the inpainting models. Please follow instructions in the notebook to set up inference with your desired configurations.

**Training**

If you are interested in training our models on custom data, please refer to the list of training configurations under the folder ```training_configs/```. To train everything from scratch, the complete model will need to go through a total of 4 training stages. Below lists the stages and their respective configuration templates:
 ```
Stage 1: training the VQGAN backbone 
	- training_configs/places_vqgan.yaml 
Stage 2: training the encoder module
	- training_configs/places_partialencoder.yaml 
Stage 3: training the transformer module
	- training_configs/places_transformer.yaml 
Stage 4: training the decoder module
	- training_configs/places_unet_256.yaml 
	- training_configs/places_unet_512.yaml 
```

Note that the modules for stage 2,3,4 can be trained independently, or concurrently,  as these stages only require a pretrained VQGAN backbone from stage 1. 

Please modify the path to the dataset, the path to the pretrained model, and optionally other hyperparameters in these configuration files to suit your needs. The basic command for training these models is as follow:
```
python train.py --base PATH_TO_CONFIG -n NAME --gpus GPU_INDEX 
```
For instance, to train the VQGAN backbone on a single gpu at index 0:
```
python train.py --base training_configs/places_vqgan.yaml -n my_vqgan_backbone --gpus 0, 
```
 To train the transformer on multiple gpus at index 1,2,3:
```
python train.py --base training_configs/places_transformer.yaml -n my_transformer --gpus 1,2,3 
```
To evaluate the trained model, please follow configuration files in ```configs/``` to modify the respective paths to each module checkpoints.

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
