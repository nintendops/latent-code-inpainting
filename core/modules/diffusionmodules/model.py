# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.modules.diffusionmodules.core_layers import Conv2dBlock as ConvBlock
import core.modules.diffusionmodules.mat as MAT
import core.modules.diffusionmodules.stylegan as StyleGAN

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class PaddingFreeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride=stride, padding=0)

    @property
    def weight(self):
        return self.conv2.weight
    

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, padding_free=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        conv_choice = PaddingFreeConv if padding_free else torch.nn.Conv2d

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_choice(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_choice(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_choice(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class PartialResnetBlock(nn.Module):
    def __init__(self, *, 
                 in_channels, 
                 out_channels=None, 
                 conv_shortcut=False, 
                 conv_choice=MAT.Conv2dLayerPartialRestrictive,
                 clamp_ratio=0.25,
                 simple_conv=True,
                 dropout):

        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        # conv_choice =  MAT.Conv2dLayerPartialRestrictive

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv_choice(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 simple_conv=simple_conv
                                 )

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_choice(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 simple_conv=simple_conv
                                 )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_choice(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 simple_conv=simple_conv
                                                 )
            else:
                self.nin_shortcut = conv_choice(in_channels,
                                                out_channels,
                                                kernel_size=1,
                                                stride=1,
                                                simple_conv=simple_conv
                                                )

    def forward(self, x, mask, clamp_ratio=None):
        h = x
        m = mask
        h = self.norm1(h)
        h = nonlinearity(h)
        h, m = self.conv1(h, m, clamp_ratio=clamp_ratio)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h, m = self.conv2(h, m, clamp_ratio=clamp_ratio)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x, m = self.conv_shortcut(x, m, clamp_ratio=clamp_ratio)
            else:
                x, m = self.nin_shortcut(x, m, clamp_ratio=clamp_ratio)

        return x+h, m



class StyleGANResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, w_dim, resolution, conv_shortcut=False,
                 dropout, temb_channels=512, padding_free=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        conv_choice = torch.nn.Conv2d
        # self.norm1 = Normalize(in_channels)
        self.conv1 = StyleGAN.SynthesisLayer(in_channels,
                                             out_channels,
                                             w_dim, 
                                             resolution,
                                             kernel_size=3,
                                             up=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        # self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_choice(out_channels,
                                 out_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_choice(in_channels,
                                                 out_channels,
                                                 kernel_size=3,
                                                 stride=1,
                                                 padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, ws, temb):
        h = x
        # h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h, ws)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        # h = self.norm2(h)
        # h = nonlinearity(h)

        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x, mask=None):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        if mask is not None:
            return x + h_, mask
        else:
            return x+h_


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t=None):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class MaskEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels + 1

        # downsampling
        self.conv_in = torch.nn.Conv2d(self.in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, masks_in):

        masks_in = masks_in.float()
        x = torch.cat([masks_in - 0.5, x * masks_in], dim=1)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        H1, W1 = masks_in.shape[-2:]
        H2, W2 = h.shape[-2:]
        mask_out = torch.nn.functional.interpolate(masks_in, scale_factor=H2/H1)
        return h, mask_out

class PartialEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, conv_choice=MAT.Conv2dLayerPartialRestrictive,
                 attn_resolutions, clamp_ratio=0.25, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, simple_conv=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels + 1
        PartialConv = conv_choice # MAT.Conv2dLayerPartialRestrictive

        self.clamp_ratio = clamp_ratio
        self.clamp_counter = 0
        self.clamp_start = 1

        # downsampling
        self.conv_in = PartialConv(self.in_channels,
                                   self.ch,
                                   kernel_size=3,
                                   stride=1,
                                   simple_conv=simple_conv)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(PartialResnetBlock(in_channels=block_in,
                                                out_channels=block_out,
                                                conv_choice=conv_choice,
                                                dropout=dropout,
                                                simple_conv=simple_conv
                                                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = PartialConv(block_in, block_in, kernel_size=3, stride=2, simple_conv=simple_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = PartialResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       conv_choice=conv_choice,
                                       dropout=dropout,
                                       simple_conv=simple_conv)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = PartialResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       conv_choice=conv_choice,
                                       dropout=dropout,
                                       simple_conv=simple_conv)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = PartialConv(block_in,
                                    2*z_channels if double_z else z_channels,
                                    kernel_size=3,
                                    stride=1,
                                    simple_conv=simple_conv)

    def update_clamp(self):
        self.clamp_counter += 1
        return self.clamp_ratio if self.clamp_counter >= self.clamp_start else 0.1

    def forward(self, x, masks_in, clamp_ratio=None):
        m = masks_in.float()
        x = torch.cat([m - 0.5, x * m], dim=1)

        # downsampling
        h, m = self.conv_in(x, m, clamp_ratio=clamp_ratio)
        hs = [h]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h, m = self.down[i_level].block[i_block](hs[-1], m, clamp_ratio=clamp_ratio)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                h, m = self.down[i_level].downsample(hs[-1], m, clamp_ratio=clamp_ratio)
                hs.append(h)

        # middle
        h = hs[-1]
        h, m = self.mid.block_1(h, m, clamp_ratio=clamp_ratio)
        h = self.mid.attn_1(h)
        h, m = self.mid.block_2(h, m, clamp_ratio=clamp_ratio)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h, m = self.conv_out(h, m, clamp_ratio=clamp_ratio)
        mask_out = m
        return h, mask_out


class MatEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,1,2,2,4), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()

        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_first = MAT.Conv2dLayerPartial(in_channels=in_channels+1, out_channels=ch, kernel_size=3, simple_conv=False)       
        self.enc_conv = nn.ModuleList()
        self.enc_conv_2 = nn.ModuleList()
        self.att_layer = 2
        in_ch_mult = (1,)+tuple(ch_mult)

        for i_level in range(self.att_layer):
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            self.enc_conv.append(
                MAT.Conv2dLayerPartial(in_channels=block_in, out_channels=block_out, kernel_size=3, down=2, simple_conv=False)
            )

        # from 64 -> 16 -> 64
        res = resolution // 2**(self.num_resolutions - self.att_layer - 1)
        dim = block_out
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 8
        window_sizes =  [8, 16, 16, 16, 8] # [8, 16, 16, 16, 8] or [2,4,4,4,2]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = MAT.PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = MAT.PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                MAT.BasicLayer(dim=dim, input_resolution=[res, res], depth=depth, num_heads=num_heads,
                               window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               downsample=merge)
            )


        for i_level in range(self.num_resolutions - self.att_layer):
            block_in = ch*in_ch_mult[self.att_layer + i_level]
            block_out = ch*ch_mult[self.att_layer + i_level]
            if i_level != self.num_resolutions - self.att_layer - 1:
                self.enc_conv_2.append(
                    MAT.Conv2dLayerPartial(in_channels=block_in, out_channels=block_out, kernel_size=3, down=2, simple_conv=False)
                )
            else:
                self.enc_conv_2.append(
                    MAT.Conv2dLayerPartial(in_channels=block_in, out_channels=block_out, kernel_size=3, simple_conv=False)
                )


        # end
        self.norm_out = Normalize(block_out)
        self.conv_out = torch.nn.Conv2d(block_out,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, images_in, masks_in):
        masks_in = masks_in.float()
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)
        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)
        x_size = x.size()[-2:]
        # mask_size = mask.size()[-2:]
        mask_out = mask

        x = MAT.feature2token(x)
        mask = MAT.feature2token(mask)
        mid = len(self.tran) // 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            else:
                x, x_size, _ = block(x, x_size, None)
                if i > mid:
                    x = x + skips[mid - i]

        x = MAT.token2feature(x, x_size).contiguous()
        mask = mask_out
        # mid
        for i, block in enumerate(self.enc_conv_2):  
            x, mask = block(x, mask)

        # end
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x, mask



###########################################################################
class mlpDecoder(nn.Module):
    def __init__(self, *, 
                    n_features = [1024]*20, 
                    nf_in = 276, 
                    nf_out = 3, 
                    activation='relu'):
        super(mlpDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        c = nf_in
        for idx, c_out in enumerate(n_features):
            block_i = ConvBlock(c, c_out, 1, 1, 0, None, activation)
            self.blocks.append(block_i)
            c = c_out
        self.last_conv = ConvBlock(c, nf_out, 1, 1, 0, None, None)
        self.blocks.append(self.last_conv)
        self.activation = nn.Tanh() # torch.sigmoid # nonlinearity # torch.sigmoid

    def forward(self, x, y=None):
        for idx, block in enumerate(self.blocks):
            if idx > 0 and idx < len(self.blocks) - 1:
                x = block(x)
            else:
                x = block(x)
        x = self.activation(x)
        if y is not None:
            x = x + y
        return x

class RestrictedUpconv2D(torch.nn.Module):
    def __init__(self, c_in, c_out, stride=2, activation=F.relu):
        super().__init__()
        if isinstance(stride, tuple):
            assert len(stride) == 2
            sx, sy = stride
        else:
            assert isinstance(stride, int)
            sx = sy = stride

        self.activation = activation
        self.mlp = torch.nn.Conv2d(c_in, c_out*sx*sy, kernel_size=1, bias=False)
        self.sx = sx
        self.sy = sy

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.activation(self.mlp(x)) # B, C*sx*sy, H, W 
        x = x.reshape(B, -1, self.sx, self.sy, H, W)
        x = x.permute(0,1,4,2,5,3).reshape(B, -1, self.sx*H, self.sy*W)
        return x

class MappingLayer(torch.nn.Module):
    def __init__(self, c_in, c_out, chs, kernel_size=3, activation=F.relu):
        super().__init__()

        assert kernel_size % 2 == 1 and kernel_size >= 3
        padding = (kernel_size - 1) // 2
        # band-limited mapping with a 3x3 convolution
        self.mapping_conv1 = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=padding)
        c_in = c_out

        # sequential 1x1 convs 
        self.mapping_conv2 = nn.ModuleList()
        for idx, ch in enumerate(chs):            
            self.mapping_conv2.append(nn.Conv2d(c_in, ch, kernel_size=1))
            c_in = ch

        self.activation = activation

    def forward(self, x):
        x = self.mapping_conv1(x)
        x = self.activation(x)
        for conv in self.mapping_conv2:
            x = conv(x)
            x = self.activation(x)
        return x

class RestrictedDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, padding_free=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks + 3
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # mapping layer
        self.mapping = MappingLayer(
            c_in = z_channels,
            c_out = block_in,
            chs = [block_in]*10,
            kernel_size = 3
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]

            if i_level != 0:
                block.append(RestrictedUpconv2D(c_in=block_in,
                                                c_out=block_out,
                                                stride=2))
            else:
                # no upsampling at last layer
                block.append(nn.Conv2d(block_in,
                                       block_out,
                                       kernel_size=1))
                block.append(nn.ReLU(inplace=True))

            for i_block in range(self.num_res_blocks):
                block.append(nn.Conv2d(block_out, block_out, kernel_size=1, stride=1))
                block.append(nn.ReLU(inplace=True))

            block_in = block_out           
            up = nn.Module()
            up.block = block
            curr_res = curr_res * 2
            self.up.append(up)
            # self.up.insert(0, up) # prepend to get consistent order

        # end
        # self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_ch,
                                  kernel_size=1,
                                  stride=1)
    
    def _map(self, z):
        return self.mapping(z)

    def _generate(self,z):
        h = z
        for block_i, block in enumerate(self.up):
            for layer_i, layer in enumerate(block.block):
                h = layer(h)
        if self.give_pre_end:
            return h
        # h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def forward(self, z, target_i_level=0):
        self.last_z_shape = z.shape
        # z to block_in
        h = self._map(z)
        feat = h if target_i_level == 0 else None
        # upsampling
        for block_i, block in enumerate(self.up):
            for layer_i, layer in enumerate(block.block):
                h = layer(h)
            if block_i == target_i_level:
                feat = h
        # end
        if self.give_pre_end:
            return h, feat
        # h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, feat

####################################################################################

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, padding_free=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        conv_choice = torch.nn.Conv2d if not padding_free else PaddingFreeConv 
        padding = 0 if padding_free else 1

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = conv_choice(z_channels,
                                   block_in,
                                   kernel_size=3,
                                   stride=1,
                                   padding=padding)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       padding_free=padding_free)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       padding_free=padding_free)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         padding_free=padding_free))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = conv_choice(block_in,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=padding)


    def _map(self, z):
        return z

    def _generate(self, z):
        h, _ = self.forward(z)
        return h

    def forward(self, z, target_i_level=0):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # print("Tensor In: Shape", z.shape)
        # import ipdb; ipdb.set_trace()

        # z to block_in
        h = self.conv_in(z)
        # print("Tensor conv_in: Shape", h.shape)
        # import ipdb; ipdb.set_trace()

        # middle
        h = self.mid.block_1(h, temb)
        # print("Tensor mid block 1: Shape", h.shape)
        # import ipdb; ipdb.set_trace()
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        # print("Tensor mid block 2: Shape", h.shape)
        # import ipdb; ipdb.set_trace()

        feat = h if target_i_level == 0 else None

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                # print(f"Tensor up block level {i_level}: Shape", h.shape)
                # import ipdb; ipdb.set_trace()
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if i_level == target_i_level:
                    feat = h

        # end
        if self.give_pre_end:
            return h, feat

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, feat

class StyleGANDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, w_dim=512, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, padding_free=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = StyleGAN.SynthesisLayer(z_channels,
                                               block_in,
                                               w_dim,
                                               curr_res,
                                               kernel_size=3,
                                               up=1)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = StyleGANResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               w_dim=w_dim,
                                               resolution=curr_res,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout,
                                               padding_free=padding_free)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = StyleGANResnetBlock(in_channels=block_in,
                                               out_channels=block_in,
                                               w_dim=w_dim,
                                               resolution=curr_res,
                                               temb_channels=self.temb_ch,
                                               dropout=dropout,
                                               padding_free=padding_free)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(StyleGANResnetBlock(in_channels=block_in,
                                                 out_channels=block_out,
                                                 w_dim=w_dim,
                                                 resolution=curr_res,
                                                 temb_channels=self.temb_ch,
                                                 dropout=dropout,
                                                 padding_free=padding_free))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def _map(self, z):
        return z

    def _generate(self, z):
        h, _ = self.forward(z)
        return h

    def forward(self, z, ws, target_i_level=0):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # print("Tensor In: Shape", z.shape)
        # import ipdb; ipdb.set_trace()

        # z to block_in
        h = self.conv_in(z, ws)
        # print("Tensor conv_in: Shape", h.shape)
        # import ipdb; ipdb.set_trace()

        # middle
        h = self.mid.block_1(h, ws, temb)

        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, ws, temb)

        feat = h if target_i_level == 0 else None

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, ws, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if i_level == target_i_level:
                    feat = h

        # end
        if self.give_pre_end:
            return h, feat

        # h = self.norm_out(h)
        # h = nonlinearity(h)
        h = self.conv_out(h)
        return h, feat



class VUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(c_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.z_in = torch.nn.Conv2d(z_channels,
                                    block_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, z):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h,z),dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

