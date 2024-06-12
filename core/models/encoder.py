import os
import copy
import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.distributions as Dist
import pytorch_lightning as pl
from train import instantiate_from_config
from core.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from core.modules.diffusionmodules.model import PartialEncoder, Encoder, Decoder, StyleGANDecoder, MatEncoder, MaskEncoder
from core.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from core.modules.diffusionmodules.mat import Conv2dLayerPartial, Conv2dLayerPartialRestrictive

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def write_images(path, image, n_row=1):
    image = ((image + 1) * 255 / 2).astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

def to_categorical(code, n_label):
    '''
        input: L length vector
        return: (L, N) onehot vectors
    '''
    assert code.max() < n_label
    onehot = torch.zeros([code.size(0), n_label], dtype=bool)
    onehot[np.arange(code.size(0)), code] = True
    return onehot.float().to(code.device)

# ENCODER MODEL
class MaskPartialEncoderModel(pl.LightningModule):
    def __init__(self,    
                 ddconfig,
                 n_embed,
                 embed_dim,
                 first_stage_config=None,            
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = PartialEncoder(**ddconfig, simple_conv=False)
        self.cls_head = Conv2dLayerPartialRestrictive(ddconfig["z_channels"], n_embed, kernel_size=1, simple_conv=False)

        # self.proj = torch.nn.Linear(n_embed, n_embed)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if first_stage_config is not None:
            self.init_first_stage_from_ckpt(first_stage_config)
    
        self.image_key = image_key

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config, initialize_encoder=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        self.quantize = self.first_stage_model.quantize
        self.quantize.train = disabled_train

    def set_first_stage_model(self, model):
        self.first_stage_model = model
        self.quantize = model.quantize

    def encode_logits(self, x, mask, clamp_ratio=None):
        h, mask_out = self.encoder(x, mask, clamp_ratio=clamp_ratio)
        logits, mask_out = self.cls_head(h, mask_out)
        return logits, mask_out

    @torch.no_grad()
    def encode(self, x, mask=None, clamp_ratio=None, return_ref=False):       
        temperature = 1.0
        # quant_z_gt, _, info_gt = self.first_stage_model.encode(x)        

        if mask is None:
            mask_in = torch.full([x.shape[0],1,x.shape[2],x.shape[3]], 1.0, dtype=torch.int32).to(x.device)
        else:
            mask_in = mask

        x = x * mask_in 
        quant_z_ref, _, info = self.first_stage_model.encode(x)       
        indices_ref = info[2].reshape(-1)
        logits, mask_out = self.encode_logits(x, mask_in, clamp_ratio=clamp_ratio)     
        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(B, -1, L)
        probs = F.softmax(logits / temperature, dim=-1)
        _, indices = torch.topk(probs, k=1, dim=-1)
        indices = indices.reshape(-1).int()
        bhwc = (quant_z_ref.shape[0],
                quant_z_ref.shape[2],
                quant_z_ref.shape[3],
                quant_z_ref.shape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            indices, shape=bhwc)

        info = (None, None, indices)

        if return_ref:
            return quant_z, None, info, mask_out, quant_z_ref, indices_ref
        elif mask is not None:
            return quant_z, None, info, mask_out
        else:
            return quant_z, None, info

    @torch.no_grad()
    def encode_to_z_first_stage(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        return quant_z, info[2].view(quant_z.shape[0], -1)

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def decode(self, quant_z):
        return self.first_stage_model.decode(quant_z)

    def forward(self, x, mask=None):

        if mask is not None:
            x = mask * x
        else:
            mask = torch.full([x.shape[0],1,x.shape[2],x.shape[3]], 1.0, dtype=torch.int32).to(x.device)

        logits, mask_out = self.encode_logits(x, mask)
        return logits, mask_out

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def shared_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        # obtain target quantized vectors 
        _, target_z_indices = self.encode_to_z_first_stage(x)
        # random mask function given by MAT paper
        mask_in = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
        logits, mask_out = self(x, mask_in)
        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(-1, L)
        logits_select = torch.masked_select(logits, mask_out.reshape(-1,1).bool()).reshape(-1, L)
        target_select = torch.masked_select(target_z_indices.reshape(-1), mask_out.reshape(-1).bool())
        loss = F.cross_entropy(logits_select, target_select)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.cls_head.parameters()),
                                  lr=lr, betas=(0.9, 0.95))
        return optimizer


    def configure_optimizers_with_lr(self, lr):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.cls_head.parameters()),
                                  lr=lr, betas=(0.9, 0.95))
        return optimizer


    def log_images(self, batch, **kwargs):

        log = dict()

        temperature = 1.0
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        quant_z, gt_z_indices = self.encode_to_z_first_stage(x)
        # mask_in = box_mask(x.shape, x.device, 0.5, det=True)
        mask_in = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
        logits, mask_out = self(x, mask_in)

        # upsampling
        B, _, H, W = x.shape
        _, _, gh, hw = mask_out.shape
        mask_out_reference = torch.round(torch.nn.functional.interpolate(mask_out.reshape(B,1,gh,hw).float(), scale_factor=H//gh))

        B, L, H, W = logits.shape
        logits = logits.permute(0,2,3,1).reshape(B, -1, L)

        # recomposing logits with gt logits
        probs = F.softmax(logits / temperature, dim=-1)
        _, indices_pred = torch.topk(probs, k=1, dim=-1)

        ######################################
        # quant_z, _, info, mask_out = self.encode(x, mask_in)
        # indices_pred = info[2]
        ######################################

        mask_out = mask_out.reshape(B, -1).int()
        indices_combined = mask_out * indices_pred.reshape(B, -1) + (1 - mask_out) * gt_z_indices
        xrec = self.decode_to_img(indices_combined.int(), quant_z.shape)

        # for comparison: encoding masked image with original encoder
        quant_z_ref, ref_z_indices = self.encode_to_z_first_stage(x * mask_in)
        indices_combined_ref = mask_out * ref_z_indices.reshape(B, -1) + (1 - mask_out) * gt_z_indices
        xrec_ref = self.decode_to_img(indices_combined_ref.int(), quant_z.shape)


        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)

        log["inputs"] = x
        log["inputs_masked"] = x * mask_in
        log["inputs_masked_reference"] = x * torch.round(mask_out_reference)
        log["reconstructions"] = xrec
        log["reconstructions_ref"] = xrec_ref

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
