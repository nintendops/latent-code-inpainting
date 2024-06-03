import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
import torch.distributions as Dist
import copy
from main import instantiate_from_config
from core.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from core.modules.diffusionmodules.model import PartialEncoder, Encoder, Decoder, StyleGANDecoder, MatEncoder, MaskEncoder
from core.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from core.modules.vqvae.quantize import GumbelQuantize
from core.modules.vqvae.quantize import EMAVectorQuantizer
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

class PartialDecoder(pl.LightningModule):
    '''
        Refinement model for the latent code transformer:
            refine given a recomposition of the inferred masked region and the original image
    '''
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 first_stage_config = None,
                 first_stage_model_type='vae', # vae | transformer
                 mask_lower = 0.25,
                 mask_upper = 0.75,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 restriction=False,        # whether a partial encoder is used
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 second_stage_refinement=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.use_refinement = second_stage_refinement   
        self.n_embed = n_embed
        # First Stage
        ##########################################
        # self.encoder = MatEncoder(**ddconfig)
        # self.encoder = MaskEncoder(**ddconfig)
        if restriction:
            ddconfig = dict(**ddconfig)
            ddconfig['conv_choice'] = Conv2dLayerPartial
            self.encoder = PartialEncoder(**ddconfig)
        else:
            self.encoder = MaskEncoder(**ddconfig)
        ##########################################
       
        self.decoder = Decoder(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.loss = instantiate_from_config(lossconfig)
        self.first_stage_model_type = first_stage_model_type
        
        if first_stage_config is not None:
            self.init_first_stage_from_ckpt(first_stage_config, initialize_decoder=ckpt_path is None)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.mask_function = box_mask
        self.mask_lower = mask_lower
        self.mask_upper = mask_upper

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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

    def init_first_stage_from_ckpt(self, config, initialize_decoder=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        if initialize_decoder:
            self.post_bottleneck_conv = copy.deepcopy(model.post_quant_conv)
            self.decoder = copy.deepcopy(model.decoder)

    def set_first_stage_model(self, model):
        self.first_stage_model = model

    def encode(self, x, mask):
        # encode the composited image
        h, mask_out = self.encoder(x, mask)
        h = self.bottleneck_conv(h)
        return h, mask_out

    def decode(self, h):
        h = self.post_bottleneck_conv(h)
        dec, _ = self.decoder(h)
        return dec

    def second_stage(self, x, mask):
        h, _ = self.encoder_2(x, mask)
        h = self.bottleneck_conv_2(h)
        h = self.post_bottleneck_conv_2(h)
        dec, _ = self.decoder_2(h)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_bottleneck_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    @torch.no_grad()
    def generate(self, batch, mask=None):
        return self(batch, mask_in=mask, return_fstg=False)

    def forward(self, batch, quant=None, mask_in=None, mask_out=None, return_fstg=True, use_noise=True, debug=False):

        input_raw = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask_in is None:
            mask = self.get_mask([input_raw.shape[0], 1, input_raw.shape[2], input_raw.shape[3]], input_raw.device)
        else:
            mask = mask_in

        input = input_raw * mask

        ###### for comparison only ################
        if self.first_stage_model_type == 'transformer':
            x_raw, quant_fstg = self.first_stage_model.forward_to_recon(batch, 
                                                                        mask=mask_out, 
                                                                        det=False, 
                                                                        return_quant=True)
        if self.first_stage_model_type == 'vae':
            x_raw, _ = self.first_stage_model(input_raw)
        
        if return_fstg:
            x_comp = mask * input_raw + (1 - mask) * x_raw
        ############################################

        if quant is None:
            if self.first_stage_model_type == 'vae':
                quant_gt, _, info = self.first_stage_model.encode(input_raw)
                if use_noise:
                    B, C, H, W = quant_gt.shape
                    # randomly replace indices from gt info
                    gt_indices = info[2]
                    prob = 0.1
                    rand_indices = torch.rand(gt_indices.shape) * (self.n_embed - 1)
                    rand_indices = rand_indices.int().to(input.device)
                    rand_mask = (torch.rand(gt_indices.shape) < prob).int().to(input.device)
                    gt_indices = rand_mask * rand_indices + (1-rand_mask) * gt_indices
                    quant = self.first_stage_model.quantize.get_codebook_entry(gt_indices.reshape(-1).int(), shape=(B, H, W, C))
                else:
                    quant = quant_gt
            else:
                quant = quant_fstg

        # _, _, codes = info
        B, C, H, W = quant.shape
        
        if mask_out is None:
            h, mask_out = self.encode(input, mask)
        else:
            h, _ = self.encode(input, mask)

        h = mask_out * h + quant * (1 - mask_out) * 0.5 + h * (1 - mask_out) * 0.5

        dec = self.decode(h)
        dec = input + (1 - mask) * dec

        if debug:
            return dec, mask, mask_out, quant * (1 - mask_out), h * (1 - mask_out)
        elif return_fstg:                
            return dec, mask, x_comp
        else:
            return dec, mask

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        # large random mask
        return torch.from_numpy(BatchRandomMask(shape[0], shape[-1])).to(device)
        # return box_mask(shape, device, 0.8, det=True)
        
    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.8, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, _ = self(batch, mask_in=mask_in, mask_out=mask_out)
        xfstg = None

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask = self(batch, mask_in=mask_in, mask_out=mask_out, return_fstg=False)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        from PIL import Image

        # We are always making assumption that the latent block is 16x16 here
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))

        x = self.get_input(batch, self.image_key)
        xrec, mask = self(batch, mask_in=mask_in, mask_out=mask_out, return_fstg=False)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.debug_log_image(x, batch_idx, tag='gt')
        self.debug_log_image(xrec, batch_idx)
        return self.log_dict

    def configure_optimizers_with_lr(self, lr):
        params = list(list(self.encoder.parameters()) + 
                 list(self.decoder.parameters()) + 
                 list(self.bottleneck_conv.parameters()) + 
                 list(self.post_bottleneck_conv.parameters()))

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_params(self):
        params = list(list(self.encoder.parameters()) + 
                 list(self.decoder.parameters()) + 
                 list(self.bottleneck_conv.parameters()) + 
                 list(self.post_bottleneck_conv.parameters()))
        return params

    def configure_optimizers(self):
        lr = self.learning_rate
        return self.configure_optimizers_with_lr(lr)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)

        # We are always making assumption that the latent block is 16x16 here
        if 'mask' in batch.keys():
            mask_in = batch['mask']
        else:
            mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))

        xrec, mask, xrec_fstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        
        log["inputs"] = x
        log["reconstructions"] = xrec
        log["masked_input"] = x * mask
        log['recon_fstg'] = xrec_fstg

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


