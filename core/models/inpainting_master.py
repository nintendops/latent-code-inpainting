import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import cv2
import numpy as np
import random
from train import instantiate_from_config
from core.modules.util import write_images, scatter_mask, box_mask, mixed_mask, BatchRandomMask
from core.modules.losses import DummyLoss
training_stages = {'vq', 
                   'encoder', 
                   'decoder', 
                   'transformer', 
                   'final',}

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def make_eval(model):
    if model is not None:
        model.eval()
        model.train = disabled_train

class InpaintingMaster(pl.LightningModule):
    '''
        vq stage: only need the VQGAN model
        encoder stage: need both a pretrained VQGAN model and an encoder model
        decoder stage: need an encoder model (VQGAN or refined) and a decoder model
        transformer stage: need an encoder model (VQGAN or refined) and a transformer model
        final stage: need the encoder, decoder and transformer models (end-to-end finetuning)
    '''
    def __init__(self,
                 stage,
                 vqmodel_config,
                 encoder_config=None,
                 decoder_config=None,
                 transformer_config=None,
                 unet_config=None,
                 encoder_choice='refined',
                 image_key="image",
                 ckpt_path=None,

                 ):
        super().__init__()

        assert stage in training_stages

        self.stage = stage
        self.image_key = image_key
        self.encoder_choice = encoder_choice

        Encoder = None
        Decoder = None
        Transformer = None
        Unet = None
        use_unet_decoder = False
        self.current_model = None
        self.helper_model = None

        if stage != 'vq':
            vqmodel_config.params.lossconfig = dict({"target": "core.modules.losses.DummyLoss"})

        # We always need a vq model in every stage
        # Initialize the VQ model
        VQModel = instantiate_from_config(vqmodel_config)
        # VQModel = VQModel.to(self.device)
        if stage != 'vq':
            make_eval(VQModel)

        # Next, instantaite the other models as necessary
        if self.stage == 'encoder' or encoder_choice == 'refined':
            assert encoder_config is not None
            Encoder = instantiate_from_config(encoder_config)
            Encoder.set_first_stage_model(VQModel)
            # Encoder = Encoder.to(self.device)

        if self.stage != 'encoder':
            if encoder_choice == 'vq':
                Encoder = VQModel
            else:
                make_eval(Encoder)

        if self.stage == 'final' and unet_config is not None:
            use_unet_decoder = not unet_config.params.freeze_firststage
            # overriding first stage config for unet instantiation
            if use_unet_decoder:
                unet_config.params.first_stage_model_type = 'decoder'
                unet_config.params.first_stage_config = decoder_config
                assert decoder_config is not None
            Unet = instantiate_from_config(unet_config)           
   
        if use_unet_decoder:
            Decoder = Unet.first_stage_model
            Decoder.set_first_stage_model(Encoder)
        elif self.stage == 'decoder' or self.stage == 'final':
            assert decoder_config is not None
            Decoder = instantiate_from_config(decoder_config)
            Decoder.set_first_stage_model(Encoder)
            # Decoder = Decoder.to(self.device)

        if self.stage == 'transformer' or self.stage == 'final':
            assert transformer_config is not None
            Transformer = instantiate_from_config(transformer_config)
            Transformer.set_first_stage_model(Encoder)
            # Transformer = Transformer.to(self.device)


        # Finally, set the current training model
        if self.stage == 'vq':
            ignore_keys = []
            self.current_model = VQModel
        elif self.stage == 'encoder':
            ignore_keys = ["VQModel"]
            self.current_model = Encoder
        elif self.stage == 'decoder':
            ignore_keys = ["VQModel", "Encoder"]
            self.current_model = Decoder
        elif self.stage == 'transformer':
            ignore_keys = ["VQModel"]
            self.current_model = Transformer
        else:
            ignore_keys = ["VQModel", "Encoder", "Transformer"]
            make_eval(Transformer)
            self.helper_model = (VQModel, Encoder, Transformer, Unet)    
            self.current_model = Decoder

        if ckpt_path is not None:            
            self.init_from_ckpt(ckpt_path, ignore_keys)

        self.set_device = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from scos functiontate_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def get_input(self, key, batch):
        x = batch[key].to(self.device)
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def forward(self, batch, mask=None, use_vq_decoder=False, simple_return=True, recomposition=False, use_unet=False):
        '''
            forward with the complete model. TODO: codes here may not work for the 512-res model
        '''
        use_vanilla = False
        VQModel, Encoder, Transformer, Unet = self.helper_model
        VQModel = VQModel.to(self.device)
        Encoder = Encoder.to(self.device)
        Transformer = Transformer.to(self.device)

        self.current_model = self.current_model.to(self.device)

        x = self.get_input(self.image_key, batch)       

        if mask is None:
            if 'mask' in batch.keys():
                mask = batch['mask']
                if mask.shape[-1] == 1:
                    mask = mask.permute(0,3,1,2).contiguous()
            else:
                # large-hole random mask following MAT
                mask = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
                # small-hole random mask following MAT
                # mask = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1], hole_range=[0,0.5])).to(x.device)

        x_gt = x        
        x = mask * x

        # encoding image
        quant_z, _, info, mask_out = Encoder.encode(x, mask)
        mask_out = mask_out.reshape(x.shape[0], -1)
        z_indices = info[2].reshape(x.shape[0], -1)

        # inferring missing codes given the downsampled mask
        z_indices_complete = Transformer.forward_to_indices(batch, z_indices, mask_out, det=False)

        # getting the features from the codebook with the indices
        B, C, H, W = quant_z.shape
        quant_z_complete = VQModel.quantize.get_codebook_entry(z_indices_complete.reshape(-1).int(), shape=(B, H, W, C))

        # decoding the features 
        dec, _ = self.current_model(batch, 
                                    quant=quant_z_complete, 
                                    mask_in=mask, 
                                    mask_out=mask_out.reshape(B, 1, H, W),
                                    return_fstg=False)

        if recomposition:
            # linear blending
            dec = mask * x + (1 - mask) * dec

        if Unet is not None and use_unet and not use_vanilla:
            Unet = Unet.to(self.device)
            dec = Unet.refine(dec, mask)

        if simple_return:
            return dec, mask
        else:
            return dec, z_indices, z_indices_complete, quant_z_complete, mask, mask_out

    # Note: for now, the final stage only trains the refined decoder assuming we have everything else pretrained
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if self.stage == 'vq' or self.stage == 'decoder':
            return self.current_model.training_step(batch, batch_idx, optimizer_idx)
        elif self.stage == 'encoder' or self.stage == 'transformer':
            return self.current_model.training_step(batch, batch_idx)
        else:
            x = self.get_input(self.image_key, batch)
            xrec, mask = self(batch)

            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.current_model.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.current_model.get_last_layer(), split="train")
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.current_model.loss(x, xrec, optimizer_idx, self.global_step,
                                                mask=mask, last_layer=self.current_model.get_last_layer(), split="train")
                self.log("train/discloss", disclosssmoothed_mask, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

    def validation_step(self, batch, batch_idx):
        if self.stage != 'final':
            return self.current_model.validation_step(batch, batch_idx)
        else:
            x = self.get_input(batch, self.image_key)
            xrec, mask = self(batch)
            aeloss, log_dict_ae = self.current_model.loss(x, xrec, 0, self.global_step,
                                                mask=mask, last_layer=self.current_model.get_last_layer(), split="val")
            discloss, log_dict_disc = self.current_model.loss(x, xrec, 1, self.global_step,
                                                mask=mask, last_layer=self.current_model.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]
            self.log("val/rec_loss", rec_loss,
                       prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/aeloss", aeloss,
                       prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict

    def configure_optimizers(self):
        return self.current_model.configure_optimizers_with_lr(self.learning_rate)

    def get_mask(self, shape, device):
        # small holes
        # mask = torch.from_numpy(BatchRandomMask(shape[0], shape[-1], hole_range=[0,0.5])).to(device)
        mask = box_mask(x.shape, x.device, 0.5, det=True).to(self.device).float()
        return mask

    @torch.no_grad()
    def log_images(self, batch, mask_in=None, **kwargs):
        if self.stage != 'final':
            return self.current_model.log_images(batch, **kwargs)
        else:
            log = dict()
            VQModel, _, _, Unet = self.helper_model
            VQModel = VQModel.to(self.device)

            x = self.get_input(self.image_key, batch)
            x = x.to(self.device)
            rst = self(batch, recomposition=False, mask=None, use_unet=False, simple_return=False)
            rec = rst[0]
            quant_z = rst[3]
            mask_in = rst[-2]
            rec_fstg = VQModel.decode(quant_z)

            # composition
            rec = mask_in * x + (1 - mask_in) * rec

            if Unet is not None:
                Unet = Unet.to(self.device)
                rec_unet = Unet.refine(rec, mask_in)
                log['recon_unet'] = rec_unet

            # log["inputs"] = x
            log["reconstructions"] = rec
            log["masked_input"] = x * mask_in
            # log['recon_fstg'] = rec_fstg
            return log



