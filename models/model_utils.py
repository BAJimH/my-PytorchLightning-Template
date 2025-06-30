import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import *
from collections import OrderedDict


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=True,
        act_type="lrelu",
        norm_type=None,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.act = activation(act_type) if act_type else None
        self.norm = norm(out_channels, norm_type) if norm_type else None

    def forward(self, x):
        out = self.conv(x)
        if self.act:
            out = self.act(out)
        if self.norm:
            out = self.norm(out)
        return out


def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i : i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


# network functions
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type="lrelu", slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace=True)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(negative_slope=slope)
    else:
        raise NotImplementedError(
            "[ERROR] Activation layer [%s] is not implemented!" % act_type
        )
    return layer


def norm(n_feature, norm_type="bn"):
    norm_type = norm_type.lower()
    if norm_type == "bn":
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError(
            "[ERROR] %s.sequential() does not support OrderedDict" % norm_type
        )
    return layer
