""" Convert .dat model into a pytorch module
"""
import struct
from torch.nn import Conv2d, MaxPool2d, PReLU, Linear
import torch
from torch.nn import Parameter
import torch.nn.functional as F
import math


class LinearChannelWise(torch.nn.Module):
    """ Do linear layer but on Channel """
    def __init__(self, in_features, out_features, bias=True):
        super(LinearChannelWise, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        bsz, c_in, w, h = input.size()
        input = input.permute(0, 2, 3, 1).contiguous()
        input = input.view([bsz * w * h, c_in])
        out = F.linear(input, self.weight, self.bias).view([bsz, w, h, self.out_features])
        return out.permute(0, 3, 1, 2).contiguous()

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def decode_int32(f, num=1):
    """ Decode `num` int from it """
    return struct.unpack('{}i'.format(num), f.read(4 * num))


def decode_float32(f, num=1):
    """ Decode one float 32 """
    return struct.unpack('{}f'.format(num), f.read(4 * num))


def decode_single_float32(f):
    return decode_float32(f)[0]


def decode_single_int32(f):
    return decode_int32(f)[0]


def decode_matrix(f):
    """ Return a tensor with (row, col) with type """
    row = decode_single_int32(f)
    col = decode_single_int32(f)
    mat_type = decode_single_int32(f)
    if mat_type % 8 == 5:
        contents = decode_float32(f, row * col)
        dtype = torch.float32
    elif mat_type % 8 == 4:
        contents = decode_int32(f, row * col)
        dtype = torch.int32
    else:
        raise ValueError('Invalid mat type')
    return torch.Tensor(contents).view([row, col]).to(dtype=dtype)


def decode_conv_layer(f):
    """ Return a `Conv2d` """
    in_channels = decode_single_int32(f)
    out_channels = decode_single_int32(f)
    bias_data = torch.Tensor(decode_float32(f, out_channels))
    kernels = [decode_matrix(f) for _ in range(in_channels * out_channels)]
    kernel_size = (kernels[0].size(0), kernels[0].size(1))
    conv_layer = Conv2d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size)

    # Initialize weight
    weight_data = torch.cat(kernels).view([in_channels, out_channels] + list(kernel_size))
    weight_data = weight_data.permute(1, 0, 2, 3)
    assert conv_layer.weight.shape == weight_data.shape
    assert conv_layer.bias.shape == bias_data.shape
    conv_layer.weight.data = weight_data
    conv_layer.bias.data = bias_data
    return conv_layer


def decode_max_pooling(f):
    """ Return MaxPool2D """
    kernel_x = decode_single_int32(f)
    kernel_y = decode_single_int32(f)
    stride_x = decode_single_int32(f)
    stride_y = decode_single_int32(f)
    return MaxPool2d(kernel_size=[kernel_x, kernel_y],
                     stride=[stride_x, stride_y])


def decode_linear_layer(f):
    """ Return a linear layer """
    bias_data = decode_matrix(f)
    bias_data = bias_data.squeeze(-1)
    weight_data = decode_matrix(f)
    linear_layer = LinearChannelWise(in_features=weight_data.size(0),
                                     out_features=weight_data.size(1))
    weight_data = weight_data.permute(1, 0)
    assert linear_layer.weight.data.shape == weight_data.shape
    assert linear_layer.bias.data.shape == bias_data.shape
    linear_layer.weight.data = weight_data
    linear_layer.bias.data = bias_data
    return linear_layer


def decode_prelu(f):
    """ Return a PReLU """
    weight_data = decode_matrix(f)
    weight_data = weight_data.squeeze(-1)
    prelu = PReLU(num_parameters=weight_data.shape.numel())
    assert prelu.weight.data.shape == weight_data.shape
    prelu.weight.data = weight_data
    return prelu


def decode_cnn(f):
    """ Return a torch.Module """
    cnn = torch.nn.Sequential()
    depths = decode_single_int32(f)
    print('Depth: {}'.format(depths))
    for layer_idx in range(depths):
        layer_type = decode_single_int32(f)
        if layer_type == 0:
            layer = decode_conv_layer(f)
        elif layer_type == 1:
            layer = decode_max_pooling(f)
        elif layer_type == 2:
            layer = decode_linear_layer(f)
        elif layer_type == 3:
            layer = decode_prelu(f)
        else:
            raise ValueError('Invalid layer type')
        cnn.add_module('layer_{}'.format(layer_idx), layer)
    return cnn

if __name__ == '__main__':
    pnet_path = '/home/yuchen/QuindiTech/OpenFace/lib/local/LandmarkDetector/model/mtcnn_detector/PNet.dat'
    with open(pnet_path, 'rb') as f:
        pnet = decode_cnn(f)
    print(pnet)
    imgs = torch.rand(1, 3, 128, 128)
    result = pnet(imgs)
    print('Input shape: ', imgs.shape)
    print('PNet output shape:', result.shape)