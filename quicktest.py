# some setup codes

import numpy as np
import glob
import os
import importlib
import yaml
import albumentations
import glob
import json
import torch
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
from torch.utils.data import random_split, DataLoader, Dataset
from omegaconf import OmegaConf
from tqdm.notebook import tqdm
from core.modules.util import box_mask, BatchRandomMask
from core.modules.losses.lpips import LPIPS
from skimage.transform import rescale, resize, downscale_local_mean
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning) 

gpu_idx = input("specify the gpu index for loading the model (e.g. 0): ")
device = torch.device(f'cuda:{gpu_idx}')

def imshow(images, titles=None, save_path=None):
    n_img = len(images)
    plt.rcParams['figure.figsize'] = [4*n_img, 4*n_img]
    
    if n_img > 1:
        fig, ax = plt.subplots(1, n_img)
        for i in range(n_img):
            if titles is not None and i < len(titles):
                ax[i].set_title(titles[i])
            ax[i].axis('off')
            ax[i].imshow(images[i])
    else:
        if titles is not None:
            plt.set_titile(titles[0])
        plt.axis('off')
        plt.imshow(images[0])

    if save_path is not None:
        plt.savefig(save_path)


def center_crop(image, s=512):
    h, w = image.shape[:2]
    if s > h or s > w:
        image = rescale(image, s/min(h,w), anti_aliasing=True)
        image = (image * 255).astype(np.uint8)
        h, w = image.shape[:2]
    ih = (h - s) // 2   
    iw = (w - s) // 2
    return image[ih:ih+s, iw:iw+s]
    
# load a single input
def preprocess(x, res=256, normalize=True):
    if normalize:
        x = x.transpose(2,0,1)
        x = (torch.from_numpy(x).float().to(device) / 127.5 - 1).unsqueeze(0)
    else:
        x = torch.from_numpy(x).float().to(device).unsqueeze(0)
    return torch.nn.functional.interpolate(x, size=(res,res))
    
def to_img(x):
    x = (x.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
    return x[0].detach().cpu().numpy()

def readmask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return mask

def read_image(path):
    image = center_crop(imread(path))
    image = preprocess(image)
    return image[:,:3]

def get_data(k, res, p=None):
    if p is None:
        p = places_val_files[k]                 
    gt = center_crop(imread(p))
    if len(gt.shape) == 2:
        gt = np.repeat(gt[...,None], 3, axis=2) 
    gt = preprocess(gt, res)
    try:
        mask_in = preprocess(readmask(os.path.join(maskfolder, f"{ids[k]}.png"))[None], res,normalize=False)
    except Exception:
        return gt, None
    return gt, mask_in

current_dir = os.path.abspath('.')
current_dir


# loading our model
os.chdir(current_dir)

from omegaconf import OmegaConf
import yaml
import torch.nn.functional as F
import importlib
from torch.utils.data import random_split, DataLoader, Dataset

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

config_path = "configs/places_inpainting.yaml"
config = OmegaConf.load(config_path)
config['data']['params']['batch_size'] = 1

model = instantiate_from_config(config.model).to(device)


# Inference code
import time
from skimage.transform import rescale, resize, downscale_local_mean

'''
candidate settings : 
1) 0.33 ratio, 0.8 degrad, 1.0t, no topk (more error prone)
2) 0.1 ratio, 0.95 degrad. 1.0t, no topk
3) 0.2 ratio, 0.9 degrad, 1.0t, no topk (default)
'''
def forward_to_indices(model, batch, z_indices, mask):
    x, c = model.get_xc(batch)
    x = x.to(device=device).float()
    c = c.to(device=device).float()
    quant_c, c_indices = model.encode_to_c(c)
    mask = model.preprocess_mask(mask, z_indices)
    r_indices = torch.full_like(z_indices, model.mask_token)
    z_start_indices = mask*z_indices+(1-mask)*r_indices      
    index_sample, probs, candidates = model.sample(z_start_indices.to(device=device), 
                               c_indices.to(device=device),
                               sampling_ratio=0.2,
                               temperature = 1,
                               sample=True,
                               temperature_degradation=0.9,
                               top_k=None,
                               return_probs = True,
                               scheduler = 'cosine',
                              )
    return index_sample, probs, candidates

# Load some example images
input_image_list = glob.glob("/home/chenh/data/test_large/*.jpg")

print(len(input_image_list))

idx = 12

x = read_image(input_image_list[idx]).to(device)

# Make some simple mask
mask = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1], hole_range=[0.0,0.5])).to(device)
# imshow([to_img(x), to_img(x*mask)])

# clamp ratio
cr = 0.25

with torch.no_grad():
    mask = torch.round(mask)    
    VQModel, Encoder, Transformer, Unet = model.helper_model
    VQModel = VQModel.to(device)
    Encoder = Encoder.to(device)
    Transformer = Transformer.to(device)

    quant_z, _, info, mask_out = Encoder.encode(x*mask, mask, clamp_ratio=cr)
    mask_out = mask_out.reshape(x.shape[0], -1)
    z_indices = info[2].reshape(x.shape[0], -1)
   
    new_batch = {'image': (x*mask).permute(0,2,3,1)}
    z_indices_complete, probs, candidates = forward_to_indices(Transformer, new_batch, z_indices, mask_out)
    B, C, H, W = quant_z.shape
    quant_z_complete = VQModel.quantize.get_codebook_entry(z_indices_complete.reshape(-1).int(), shape=(B, H, W, C))   
    rec_fstg = VQModel.decode(quant_z_complete)
    dec, _, mout, f0, f1 = model.current_model(new_batch, 
                                quant=quant_z_complete, 
                                mask_in=mask, 
                                mask_out=mask_out.reshape(B, 1, H, W),
                                return_fstg=False, debug=True)  
    rec = x * mask + dec * (1-mask) 
    if Unet is not None:
        Unet = Unet.to(device)
        rec = Unet.refine(rec, mask)

print(rec.shape)
# imshow([to_img(x*mask),to_img(rec)], titles=['input', 'prediction'])