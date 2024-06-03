import torch
import torch.nn.functional as F
from torch import nn
import functools

################################### core layer ####################################
def normalization(type):
    if type == 'batch2d':
        return functools.partial(torch.nn.BatchNorm2d)
    elif type == 'batch3d':
        return functools.partial(torch.nn.BatchNorm3d)
    elif type == 'inst2d':
        return functools.partial(torch.nn.InstanceNorm2d)
    elif type == 'inst3d':
        return functools.partial(torch.nn.InstanceNorm3d)
    elif type == 'spectral':
        return torch.nn.utils.spectral_norm
    elif type == 'none' or type is None:
        return functools.partial(torch.nn.Identity)
    else:
        raise NotImplementedError('Normalization {} is not implemented'.format(type))


def non_linearity(type):
    if type == 'relu':
        return functools.partial(torch.nn.ReLU)
    elif type == 'lrelu':
        return functools.partial(torch.nn.LeakyReLU, negative_slope=0.2)
    elif type == 'elu':
        return functools.partial(torch.nn.ELU)
    elif type == 'none' or type is None:
        return functools.partial(torch.nn.Identity)
    else:
        raise NotImplementedError('Nonlinearity {} is not implemented'.format(type))


class StandardBlock(nn.Module):
    def __init__(self, type, nf_in, nf_out, kernel_size, stride, padding, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, padding_mode='zeros'):
        super().__init__()

        # shortcut if features have different size
        self.n_features_in = nf_in
        self.n_features_out = nf_out

        self.norm_type = norm_type
        self.bias = bias
        norm = normalization(norm_type)
        activation = non_linearity(activation_type)

        if type == 'linear':
            layer = torch.nn.Linear
            dropout = torch.nn.Dropout
            self.layer = layer(nf_in, nf_out, bias)

        elif type == 'conv_2d':
            layer = torch.nn.Conv2d
            dropout = torch.nn.Dropout2d
            self.layer = layer(nf_in, nf_out, kernel_size, stride, padding, bias, padding_mode=padding_mode)

        elif type == 'conv_transpose_2d':
            layer = torch.nn.ConvTranspose2d
            dropout = torch.nn.Dropout2d
            self.layer = layer(nf_in, nf_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, padding_mode=padding_mode)

        elif type == 'conv_3d':
            layer = torch.nn.Conv3d
            dropout = torch.nn.Dropout3d
            self.layer = layer(nf_in, nf_out, kernel_size, stride, padding, bias, padding_mode=padding_mode)

        elif type == 'conv_transpose_3d':
            layer = torch.nn.ConvTranspose3d
            dropout = torch.nn.Dropout3d
            self.layer = layer(nf_in, nf_out, kernel_size, stride, padding, biaspadding_mode=padding_mode)

        else:
            raise NotImplementedError

        self.activation = activation()
        self.dropout = dropout(dropout_ratio)

        if self.norm_type == 'specular':
            self.norm = norm(self.conv)
        else:
            self.norm = norm(nf_out)

    def forward(self, input, style=None):

        if self.norm_type == 'specular':
            output = self.norm(input)

        elif self.norm_type == 'adain':
            output = self.layer(input)
            output = self.norm(output, style)
        else:
            output = self.layer(input)

            # output = self.norm(output)

        output = self.activation(output)
        output = self.dropout(output)

        return output


class LinearBlock(StandardBlock):
    def __init__(self, nf_in, nf_out, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, ):
        super().__init__('linear', nf_in, nf_out, kernel_size=None, stride=None, padding=None, norm_type=norm_type, activation_type=activation_type,
                         dropout_ratio=dropout_ratio, bias=bias)


class Conv2dBlock(StandardBlock):
    def __init__(self, nf_in, nf_out, kernel_size, stride, padding, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, padding_mode='zeros'):
        super().__init__('conv_2d', nf_in, nf_out, kernel_size, stride, padding, norm_type, activation_type, dropout_ratio, bias, padding_mode=padding_mode)


class ConvTrans2dBlock(StandardBlock):
    def __init__(self, nf_in, nf_out, kernel_size, stride, padding, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, padding_mode='zeros'):
        super().__init__('conv_transpose_2d', nf_in, nf_out, kernel_size, stride, padding, norm_type, activation_type, dropout_ratio, bias, padding_mode=padding_mode)


class Conv3dBlock(StandardBlock):
    def __init__(self, nf_in, nf_out, kernel_size, stride, padding, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, padding_mode='zeros'):
        super().__init__('conv_3d', nf_in, nf_out, kernel_size, stride, padding, norm_type, activation_type, dropout_ratio, bias, padding_mode=padding_mode)


class ConvTrans3dBlock(StandardBlock):
    def __init__(self, nf_in, nf_out, kernel_size, stride, padding, norm_type=None, activation_type=None, dropout_ratio=0.0, bias=True, padding_mode='zeros'):
        super().__init__('conv_transpose_3d', nf_in, nf_out, kernel_size, stride, padding, norm_type, activation_type, dropout_ratio, bias, padding_mode=padding_mode)
