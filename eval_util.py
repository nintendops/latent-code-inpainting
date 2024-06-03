# load config for transformer
import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import yaml
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import time
import cv2
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset
from core.models.cond_transformer import Net2NetTransformer


def write_images(path, image):
    image = np.clip(image, -1, 1)
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

def write_image_grid(path, image, n_rows=1):
    image = torchvision.utils.make_grid(image, nrow=n_rows)
    image = image.detach().cpu().numpy().transpose(1,2,0)
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

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


def eval_fixsample(batch, image_idx, model, config, n_sample=4, do_not_generate=False):
    '''
        vanilla evaluation code: generate from a fixed size code map

    '''
    print(f"Generating samples for batch data {image_idx}")

    # ---------------------- first log the input data -----------------------------
    image = batch['image']
    if unconditional:
        segmentation = image
    else:
        segmentation = batch['segmentation']
        write_images(os.path.join(save_path, f'image_{image_idx}_segmentation.png'), segmentation)
    write_images(os.path.join(save_path, f'image_{image_idx}_src.png'), image)
    # -----------------------------------------------------------------------------

    if do_not_generate:
        return

    tensify = lambda x: torch.from_numpy(x[None]).permute(0,3,1,2).contiguous().float().to(device)
    tensor_to_numpy = lambda x:x.detach().cpu().numpy()[0].transpose(1,2,0)
    seg_tensor = tensify(segmentation)

    # ---------------------- dealing with conditional code -----------------------------
    c_code, c_indices = model.encode_to_c(seg_tensor)
    # if not unconditional:
    #     seg_rec = model.cond_stage_model.decode(c_code)
    #     seg_rec = F.softmax(seg_rec,dim=1)
    # ----------------------------------------------------------------------------------

    nb = c_code.shape[0]
    codebook_size = config.model.params.first_stage_config.params.embed_dim
    res = 256
    c_code_res = int(res**0.5)
    z_indices_shape = [nb,res]
    z_code_shape = [nb, codebook_size,c_code_res, c_code_res]

    # using random codebook entries
    z_indices = torch.randint(codebook_size, z_indices_shape, device=model.device)

    # inpainting setting: replace random z indices with that of an encoded batch image
    x = tensify(batch['image'])
    c = x
    _, z_indices = model.encode_to_z(x)
    _, c_indices = model.encode_to_c(c)

    if not unconditional:
        cidx = c_indices
        cidx = cidx.reshape(c_code.shape[0],c_code.shape[2],c_code.shape[3])

    temperature = 1.0
    top_k = 100
    update_every = 100000
    start_t = time.time()
    start_i = 0
    start_j = 0

    for sample in range(n_sample):        
        idx = z_indices
        idx = idx.reshape(z_code_shape[0],z_code_shape[2],z_code_shape[3])
        for i in range(start_i, z_code_shape[2]-0):
          if i <= 8:
            local_i = i
          elif z_code_shape[2]-i < 8:
            local_i = 16-(z_code_shape[2]-i)
          else:
            local_i = 8
          for j in range(start_j, z_code_shape[3]-0):
            if j <= 8:
              local_j = j
            elif z_code_shape[3]-j < 8:
              local_j = 16-(z_code_shape[3]-j)
            else:
              local_j = 8
            
            i_start = i-local_i
            i_end = i_start+16
            j_start = j-local_j
            j_end = j_start+16
            # print(i_start, i_end, j_start, j_end)
            patch = idx[:,i_start:i_end,j_start:j_end]
            patch = patch.reshape(patch.shape[0],-1)
            if unconditional:
                patch = torch.cat((c_indices, patch), dim=1)
                logits,_ = model.transformer(patch[:,:-1])
            else:
                cpatch = cidx[:, i_start:i_end, j_start:j_end]
                cpatch = cpatch.reshape(cpatch.shape[0], -1)
                # patch = torch.cat((cpatch, patch), dim=1)
                logits,_ = model.transformer(patch[:,:-1], cpatch)

            logits = logits[:, -256:, :]
            logits = logits.reshape(z_code_shape[0],16,16,-1)
            logits = logits[:,local_i,local_j,:]
            logits = logits/temperature
            
            if top_k is not None:
              logits = model.top_k_logits(logits, top_k)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx[:,i,j] = torch.multinomial(probs, num_samples=1)
            step = i*z_code_shape[3]+j
            if step==0 or step==z_code_shape[2]*z_code_shape[3]-1:
              x_sample = model.decode_to_img(idx, z_code_shape)
              print(f"Time: {time.time() - start_t} seconds")
              print(f"Sample: {sample} | Step: ({i},{j}) | Local: ({local_i},{local_j}) | Crop: ({i_start}:{i_end},{j_start}:{j_end})")
              write_images(os.path.join(save_path, f"image_{image_idx}_sample_{sample}.png"), tensor_to_numpy(x_sample))

def tensify(x, device, permute=True):
    if permute:
        return torch.from_numpy(x[None]).permute(0,3,1,2).contiguous().float().to(device)
    else:
        return torch.from_numpy(x[None]).contiguous().float().to(device)

def tensor_to_numpy(x):
    return x.detach().cpu().numpy()[0].transpose(1,2,0)

def eval_vqgan(*, data, idx, model, opt, config, save_path, **ignorekwargs):
    split_generate = False # opt.split
    # ---------------------- first log the input data -----------------------------
    image = data['image']
    write_images(os.path.join(save_path, f'{idx}_gt_image.png'), image)
    # -----------------------------------------------------------------------------
  
    # encoding
    x = tensify(image, torch.device('cuda'))
    quant, probs, info = model.encode(x)

    # -----------------------------------------------------------------------------
    #  quant shape: B, C, H, W
    # B, C, H, W = quant.shape
    # _, _, HH, WW = x.shape
    # s = HH // H
    # if split_generate:
    #     split_factor = 2
    #     target_image = np.zeros([HH, WW, 3], dtype=np.float32)
    #     _quant = model.post_quant_conv(quant)
    #     quant_mapped = model.decoder._map(_quant)
    #     for i in range(0, H, H//split_factor):
    #         for j in range(0, W, W//split_factor):
    #             patch_quant = quant_mapped[:,:,i:i+H//split_factor, j:j+W//split_factor]
    #             patch_recon = tensor_to_numpy(model.decoder._generate(patch_quant))
    #             target_image[s*i:s*i+s*H//split_factor, s*j:s*j+s*W//split_factor] = patch_recon
    #     write_images(os.path.join(save_path, f'{idx}_split_image.png'), np.clip(target_image, -1.0, 1.0))
    # -------------------------------------------------------------------------------

    # decode without splitting code map anyway
    recon = tensor_to_numpy(model.decode(quant))
    # recon = tensor_to_numpy(model.decode(quant))
    write_images(os.path.join(save_path, f'{idx}_recon_image.png'), np.clip(recon, -1.0, 1.0))
    return 


def eval_attgan(*, data, idx, model, opt, config, save_path, **ignorekwargs):
    split_generate = opt.split
    # ---------------------- first log the input data -----------------------------
    image = data['image']
    write_images(os.path.join(save_path, f'{idx}_gt_image.png'), image)
    # -----------------------------------------------------------------------------
   
    # encoding
    x = tensify(image, torch.device('cuda'))
    quant, probs, info = model.encode(x)

    #  quant shape: B, C, H, W
    B, C, H, W = quant.shape
    _, _, HH, WW = x.shape
    s = HH // H

    if split_generate:
        split_factor = 2
        target_image = np.zeros([HH, WW, 3], dtype=np.float32)
        _quant = model.post_quant_conv(quant)
        quant_mapped = model.decoder._map(_quant)
        for i in range(0, H, H//split_factor):
            for j in range(0, W, W//split_factor):
                patch_quant = quant_mapped[:,:,i:i+H//split_factor, j:j+W//split_factor]
                patch_recon = tensor_to_numpy(model.decoder._generate(patch_quant))
                target_image[s*i:s*i+s*H//split_factor, s*j:s*j+s*W//split_factor] = patch_recon
        write_images(os.path.join(save_path, f'{idx}_split_image.png'), np.clip(target_image, -1.0, 1.0))

    # decode without splitting code map anyway
    recon = tensor_to_numpy(model.decode(quant))
    write_images(os.path.join(save_path, f'{idx}_recon_image.png'), np.clip(recon, -1.0, 1.0))

    quant, probs, info = model.encode(x, use_topk=True)
    recon_topk = tensor_to_numpy(model.decode(quant))
    write_images(os.path.join(save_path, f'{idx}_recon_quant_image.png'), np.clip(recon_topk, -1.0, 1.0))

    return 

def LowerHalfMask(s):
    mask = np.ones((s, s), np.uint8)
    mask[s//2:] = 0
    return mask[..., np.newaxis].astype(np.float32)

def eval_half(*, data, idx, model, opt, config, save_path, **ignorekwargs):
    batch = data
    image_idx = idx
    transformer = model
    n_sample = opt.sample 
    split_generate = opt.split

    only_crop_tokens = False
    
    # ---------------------- first log the input data -----------------------------
    image = batch['image']
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    if unconditional:
        segmentation = image
    else:
        segmentation = batch['segmentation']
        write_images(os.path.join(save_path, f'image_{image_idx}_segmentation.png'), segmentation)
    write_images(os.path.join(save_path, f'image_{image_idx}_src.png'), image)
    # -----------------------------------------------------------------------------

    codebook_size = config.model.params.first_stage_config.params.embed_dim
    nb = 1
    res = 256
    c_code_res = 16
    
    H, W, C = image.shape

    # mask out half of the image before the encoding
    mask = LowerHalfMask(H)

    if only_crop_tokens:
        write_images(os.path.join(save_path, f'image_{image_idx}_masked.png'), image * mask)
    else:
        image = image * mask
        write_images(os.path.join(save_path, f'image_{image_idx}_masked.png'), image)

    # get the gt z_indices first
    x = tensify(image, torch.device('cuda')) # B, C, H, W
    _, z_indices = model.encode_to_z(x)

    # # use conditional inpaiting in all cases?
    # c = tensify(segmentation, torch.device('cuda'))
    # _, c_indices = model.encode_to_c(c)

    for i_sample in range(n_sample):
        print(f"Generating samples for batch data {image_idx} at sample #{i_sample}")

        # ---------------------- generate image code -----------------------------
        z_indices = generate(transformer, 
                             1, 
                             codebook_size, 
                             c_code_res,
                             inpainting_multiplier=(0.5, 1), 
                             gt_z_indices=z_indices, 
                             c_indices=None)
        # ------------------------------------------------------------------------

        # ---------------------- decode code blocks into image ----------------------------- 
        new_z_code_shape = [nb, codebook_size, c_code_res, c_code_res]
        x_sample = transformer.decode_to_img(z_indices, new_z_code_shape).detach().cpu()
        # x_sample = F.softmax(x_sample, dim=1)
        target_image = x_sample[0].numpy().transpose(1,2,0)

        # create grid image
        write_images(os.path.join(save_path, f'image_{image_idx}_sample_{i_sample}_generate.png'), target_image)
        # ---------------------------------------------------------------------------------


def eval_transformer_log(*, data, idx, model, opt, config, save_path, log_input=False, **ignorekwargs):
    batch = data
    image_name = batch['filename_']
    image_idx = idx
    # transformer = model
    n_sample = opt.sample 
    split_generate = opt.split
    
    # ---------------------- first log the input data -----------------------------
    image = batch['image']
    unconditional = True # config.model.params.cond_stage_config == "__is_unconditional__"
    
    if unconditional:
        segmentation = image
    else:
        segmentation = batch['segmentation']
        if log_input:
            write_images(os.path.join(save_path, f'image_{image_idx}_segmentation.png'), segmentation)
    if log_input:
        write_images(os.path.join(save_path, f'image_{image_idx}_src.png'), image)
    # -----------------------------------------------------------------------------

    # codebook_size = config.model.params.first_stage_config.params.embed_dim
    # nb = 1
    # res = 256
    # c_code_res = 16
    # H, W, C = image.shape

    true_batch = dict()
    true_batch['image'] = tensify(image, torch.device('cuda'), permute=False)
    true_batch['segmentation'] = tensify(segmentation, torch.device('cuda'), permute=False) 

    if 'mask' in batch.keys():
        true_batch['mask'] = tensify(batch['mask'], torch.device('cuda'), permute=True)

    for i_sample in range(n_sample):
        print(f"Generating samples for {image_name} at sample #{i_sample}")
        log_images = model.log_images(true_batch)
        for k in log_images.keys():
            target_image = log_images[k][0].cpu().numpy().transpose(1,2,0)
            write_images(os.path.join(save_path, f'{image_name}_sample_{i_sample}_{k}.png'), target_image)


def eval_mult(*, data, idx, model, opt, config, save_path, **ignorekwargs):

    batch = data
    image_idx = idx
    transformer = model
    multiplier = opt.multiplier
    n_sample = opt.sample 
    split_generate = opt.split
    inpainting = opt.inpainting
    do_not_generate = False

    # ---------------------- first log the input data -----------------------------
    image = batch['image']
    unconditional = config.model.params.cond_stage_config == "__is_unconditional__"
    if unconditional:
        segmentation = image
    else:
        segmentation = batch['segmentation']
        write_images(os.path.join(save_path, f'image_{image_idx}_segmentation.png'), segmentation)
    write_images(os.path.join(save_path, f'image_{image_idx}_src.png'), image)
    # -----------------------------------------------------------------------------

    if do_not_generate:
        return

    codebook_size = config.model.params.first_stage_config.params.embed_dim
    nb = 1
    res = 256
    c_code_res = 16

    if inpainting:
        # inpainting setting: replace random z indices with that of an encoded batch image
        x = tensify(image, torch.device('cuda'))
        _, z_indices = model.encode_to_z(x)
    else:
        z_indices = None

    # use conditinal inpaiting in all cases
    c = tensify(segmentation, torch.device('cuda'))
    _, c_indices = model.encode_to_c(c)

    if not unconditional:
        codebook_size_cond = config_cond.model.params.first_stage_config.params.embed_dim
        # seg_tensor = tensify(segmentation)
        # c_code, c_indices = transformer.encode_to_c(seg_tensor)
        # ---------------------- generate conditional code -----------------------------
        # (nb, nnx, nny)
        c_indices_large = generate(transformer_cond, multiplier, codebook_size_cond, c_code_res, gt_z_indices=c_indices)
        # reconstruct the conditional shape
        new_c_code_shape = [nb, codebook_size_cond, multiplier*c_code_res, multiplier*c_code_res]
        seg_rec = transformer_cond.decode_to_img(c_indices_large, new_c_code_shape)
        seg_rec = F.softmax(seg_rec,dim=1)
        # seg_rec = seg_rec[0].detach().cpu().numpy().transpose(1,2,0)
        # write_images(os.path.join(save_path, f'image_{image_idx}_seg_sample_{i_sample}_generate.png'), seg_rec)
    else:
        c_indices_large = None

    for i_sample in range(n_sample):
        print(f"Generating samples for batch data {image_idx} at sample #{i_sample}")

        # ---------------------- generate image code -----------------------------
        z_indices_large = generate(transformer, 
                                   multiplier, 
                                   codebook_size, 
                                   c_code_res, 
                                   gt_z_indices=z_indices, 
                                   c_indices=c_indices_large)
        # ------------------------------------------------------------------------

        # ---------------------- decode code blocks into image ----------------------------- 
        if split_generate:
            z_code_shape = [nb, codebook_size, c_code_res, c_code_res]
            target_image = np.zeros([multiplier*res, multiplier*res, 3])
            _, nnx, nny = z_indices_large.shape
            for i in range(0, nnx, c_code_res):
                for j in range(0, nny, c_code_res):
                    patch_code = z_indices_large[:, i:i+c_code_res, j:j+c_code_res]
                    x_sample = transformer.decode_to_img(patch_code, z_code_shape)
                    # x_sample = F.softmax(x_sample, dim=1)
                    patch_image = x_sample[0].detach().cpu().numpy().transpose(1,2,0)
                    target_image[i*c_code_res:i*c_code_res+res, j*c_code_res:j*c_code_res+res] = patch_image
            x_sample = torch.from_numpy(target_image.transpose(2,0,1))
        else:
            new_z_code_shape = [nb, codebook_size, multiplier*c_code_res, multiplier*c_code_res]
            x_sample = transformer.decode_to_img(z_indices_large, new_z_code_shape).detach().cpu()
            # x_sample = F.softmax(x_sample, dim=1)
            target_image = x_sample[0].numpy().transpose(1,2,0)

        # create grid image
        if unconditional:
            write_images(os.path.join(save_path, f'image_{image_idx}_sample_{i_sample}_generate.png'), target_image)
        else:
            output_image = torch.cat([x_sample, seg_rec.detach().cpu()],dim=0)
            write_image_grid(os.path.join(save_path, f'image_{image_idx}_sample_{i_sample}_generate.png'), output_image, n_rows=2)
        # write_images(os.path.join(save_path, f'image_{image_idx}_sample_{i_sample}_generate.png'), target_image)
        # ---------------------------------------------------------------------------------

def generate(model, 
             multiplier, 
             codebook_size, 
             res, 
             inpainting_multiplier=(1,1), 
             image_res=256, 
             gt_z_indices=None, 
             c_indices=None):
    '''
     generate a multiplied-scale scene with an arbitrary-sized random codebook
     each 16x16 code block is decoded into a 256x256 patch of pixels
        - nx, ny: dim of codebook to pre-fill
        - target_size (nnx, nny): dim of codebook to generate
        - idx: the codebook block representing the large-scale image to be generated (of dimension nnx, nny)

     return: an inferred codebook of size (nb, nnx, nny)
    '''
    # multiplier = 4
    nx = res
    ny = res
    i_nx = int(inpainting_multiplier[0] * nx)
    i_ny = int(inpainting_multiplier[1] * ny)
    nnx = res * multiplier
    nny = res * multiplier
    nb = 1
    temperature = 1.0
    top_k = 100

    # block size for each generation step (16x16) and step size of the sliding windows (8)
    c_code_res = 16
    step_size = 8

    # target image to be filled
    target_image = np.zeros([multiplier*image_res, multiplier*image_res, 3])

    # randomly initialized codebook to be inferred iteratively
    z_indices = torch.randint(codebook_size, [nb, nnx, nny], device=model.device)
    occupancy = np.zeros(z_indices.shape).astype(bool)
    
    # (inpainting) partially fill the z_indices with known data
    if gt_z_indices is not None:
        z_indices[:,:i_nx, :i_ny] = (gt_z_indices.reshape(nb, nx, ny))[:,:i_nx, :i_ny]
        occupancy[:,:i_nx, :i_ny] = True

    # getting the dummy conditional code for unconditional generation
    if c_indices is None:
        c = torch.ones(nb, 1)*0
        c_idx = c.long().to(model.device)

    start_t = time.time()

    # outer loop: dividing the codebook into 16x16 blocks, with a moving window sliding at 8 blocks per step
    for i in range(0, nnx - step_size, step_size):
        for j in range(0, nny - step_size, step_size):
            idx = z_indices[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)
            occ = occupancy[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)       

            if c_indices is not None:
                c_idx = c_indices[:, i:i+c_code_res, j:j+c_code_res].reshape(nb, -1)

            # only update a block in the codebook if it is not occupied
            for ii in range(idx.shape[1]):
                if not occ[0,ii]:
                    if c_indices is not None:
                        logits,_ = model.transformer(idx[:,:ii], c_idx)
                    else:
                        patch = torch.cat((c_idx, idx[:,:ii]),1)
                        logits,_ = model.transformer(patch)
                    logits = logits[:, -1, :]
                    logits = logits/temperature
                    if top_k is not None:
                      logits = model.top_k_logits(logits, top_k)
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    idx[:,ii] = torch.multinomial(probs, num_samples=1)
                    occ[:,ii] = True
            idx = idx.reshape(nb, c_code_res, c_code_res)
            occ = occ.reshape(nb, c_code_res, c_code_res)
            z_indices[:, i:i+c_code_res, j:j+c_code_res] = idx
            occupancy[:, i:i+c_code_res, j:j+c_code_res] = occ
            print(f"Time: {time.time() - start_t} seconds")
            print(f"Step: ({i},{j})")

    return z_indices

def load_config(config_path):
    config = OmegaConf.load(config_path)
    config ['data']['params']['batch_size'] = 4
    return config
