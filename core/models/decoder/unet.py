import os
import torch
import torch.nn.functional as F
import torch.distributions as Dist
import pytorch_lightning as pl
import cv2
import numpy as np
import random
import copy
from main import instantiate_from_config
from core.modules.util import scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
from core.modules.diffusionmodules.model import PartialEncoder, Encoder, Decoder, MatEncoder, MaskEncoder
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


class RefinementUNet(pl.LightningModule):
    '''
        Refinement model for the decoder:
            refine a recomposed image
    '''
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 edconfig = None,
                 input_res = 256,
                 lossconfig = None,
                 first_stage_config = None,
                 first_stage_model_type='vae', # vae | transformer
                 mask_lower = 0.25,
                 mask_upper = 0.75,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 freeze_firststage=True
                 ):
        super().__init__()
        self.image_key = image_key
        self.input_res = input_res

        if edconfig is None:
            edconfig = ddconfig

        self.encoder = Encoder(**edconfig)       
        self.decoder = Decoder(**ddconfig)
        self.bottleneck_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_bottleneck_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.freeze_firststage = freeze_firststage

        if lossconfig is not None:
            self.loss = instantiate_from_config(lossconfig)
        else:
            self.loss = None

        self.first_stage_model_type = first_stage_model_type

        if ckpt_path is not None and self.freeze_firststage:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        if first_stage_config is not None and (ckpt_path is None or self.freeze_firststage):
            print("Initializing with a pretrained first stage model")
            # initialize the U-Net with vq-vae if not resumed from a checkpoint
            self.init_first_stage_from_ckpt(first_stage_config, initialize_current=False)

        if first_stage_config is not None and ckpt_path is not None and not self.freeze_firststage:
            print("Initializing with a pretrained U-Net model")
            self.first_stage_model = instantiate_from_config(first_stage_config)
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if not self.freeze_firststage:
            self.loss_fstg = self.first_stage_model.loss

        self.image_key = image_key

        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        self.mask_function = box_mask
        self.mask_lower = mask_lower
        self.mask_upper = mask_upper


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

    def init_first_stage_from_ckpt(self, config, initialize_current=False):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train

        self.first_stage_model = model

        if self.first_stage_model_type == 'vae':
            target_model = model
        elif self.first_stage_model_type == 'transformer':
            target_model = model.first_stage_model
        elif self.first_stage_model_type == 'decoder':
            target_model = model.first_stage_model
        else:
            raise Exception(f"Unrecognized model type {self.first_stage_model_type}")

        if initialize_current:
            self.encoder = copy.deepcopy(target_model.encoder)
            self.bottleneck_conv = copy.deepcopy(target_model.quant_conv)
            self.post_bottleneck_conv = copy.deepcopy(target_model.post_quant_conv)
            self.decoder = copy.deepcopy(target_model.decoder)

    def set_first_stage_model(self, model):
        self.first_stage_model = model

    def encode(self, x):
        # encode the composited image
        h = self.encoder(x)
        h = self.bottleneck_conv(h)
        return h

    def decode(self, h):
        h = self.post_bottleneck_conv(h)
        dec, _ = self.decoder(h)
        return dec

    def decode_at_layer(self, quant, i):
        quant = self.post_bottleneck_conv(quant)
        _, feat = self.decoder(quant, target_i_level = i)
        return feat

    @torch.no_grad()
    def refine(self, img, mask, recomp=True):
        h = self.encode(img)
        dec = self.decode(h)
        if recomp:
            dec = mask * img + (1 - mask) * dec    
        return dec


    def forward(self, batch, quant=None, mask_in=None, mask_out=None, return_fstg=True, debug=False):

        input_raw = self.get_input(batch, self.image_key)

        # first, get a composition of quantized reconstruction and the original image
        if mask_in is None:
            mask = self.get_mask([input.shape[0], 1, input.shape[2], input.shape[3]], input.device)
        else:
            mask = mask_in

        input = input_raw * mask

        # rescale to specified input resolution
        input_lr = torch.nn.functional.interpolate(input, (self.input_res, self.input_res))
        mask_lr = torch.nn.functional.interpolate(mask, (self.input_res, self.input_res))
        
        if self.first_stage_model_type == 'transformer':
            x_raw, quant_fstg = self.first_stage_model.forward_to_recon(batch, 
                                                                        mask=mask_lr, 
                                                                        det=False, 
                                                                        return_quant=True)    
            x_comp = input_lr + (1 - mask_lr) * x_raw
        elif self.first_stage_model_type == 'vae':
            x_raw, _ = self.first_stage_model(input_lr)
            x_comp = input_lr + (1 - mask_lr) * x_raw
        elif self.first_stage_model_type == 'decoder':
            if self.freeze_firststage:
                # this create a lr mask since we have scaled the batch image
                x_raw, _ = self.first_stage_model.generate(batch, rescale=self.input_res, mask_in=mask_lr, recomposition=False)
            else:
                x_raw, _ = self.first_stage_model(batch, rescale=self.input_res, mask_in=mask_lr, recomposition=False, return_fstg=False)
            x_comp = input_lr + (1 - mask_lr) * x_raw
        else:
            raise Exception(f"Unrecognized model type {self.first_stage_model_type}")

        # forward pass with the recomposed image
        h = self.encode(x_comp)
        dec = self.decode(h)
        dec = input + (1 - mask) * dec
        return dec, mask, x_comp

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float().to(self.device)

    def get_mask(self, shape, device):
        # return box_mask(shape, device, 0.8, det=True)
        return torch.from_numpy(BatchRandomMask(shape[0], shape[-1])).to(device)
        
    def get_mask_eval(self, shape, device):
        return box_mask(shape, device, 0.5, det=True)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # We are always making assumption that the latent block is 16x16 here
        x = self.get_input(batch, self.image_key)
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        # mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, x_fstg = self(batch, mask_in=mask_in, mask_out=None)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step,
                                            mask=mask, last_layer=self.get_last_layer(), split="train")

            if not self.freeze_firststage:
                x_lr = torch.nn.functional.interpolate(x, (self.input_res, self.input_res)) 
                mask_lr = torch.nn.functional.interpolate(mask, (self.input_res, self.input_res)) 
                
                aeloss_fstg, log_dict_ae_fstg = self.loss_fstg(x_lr, x_fstg, optimizer_idx, self.global_step,
                                                               mask=mask_lr, last_layer=self.first_stage_model.get_last_layer(), split="train")
                aeloss = aeloss + aeloss_fstg

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.get_last_layer(), split="train")
            if not self.freeze_firststage:
                x_lr = torch.nn.functional.interpolate(x, (self.input_res, self.input_res)) 
                mask_lr = torch.nn.functional.interpolate(mask, (self.input_res, self.input_res)) 
                discloss_fstg, log_dict_disc_fstg = self.loss_fstg(x_lr, x_fstg, optimizer_idx, self.global_step,
                                                                   mask=mask_lr, last_layer=self.first_stage_model.get_last_layer(), split="train")
                discloss = discloss + discloss_fstg

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

    def configure_optimizers_with_lr(self, lr):
        params = list(list(self.encoder.parameters()) + 
                 list(self.decoder.parameters()) + 
                 list(self.bottleneck_conv.parameters()) + 
                 list(self.post_bottleneck_conv.parameters()))

        if not self.freeze_firststage:
            params_fstg = self.first_stage_model.get_params()
            params += params_fstg

        opt_ae = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.9))

        disc_params = list(self.loss.discriminator.parameters())

        if not self.freeze_firststage:
            disc_params += list(self.loss_fstg.discriminator.parameters())

        opt_disc = torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []


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
        mask_in = self.get_mask([x.shape[0], 1, x.shape[-2], x.shape[-1]], x.device).float()
        mask_out = torch.round(torch.nn.functional.interpolate(mask_in, scale_factor=16/x.shape[-1]))
        xrec, mask, xrec_fstg = self(batch, mask_in=mask_in, mask_out=mask_out)
        
        # log["inputs"] = x
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
