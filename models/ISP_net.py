import os
import sys
import cv2
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from collections import OrderedDict
from pytorch_msssim import ssim

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .model_utils import *
from torch.nn import MultiheadAttention

from torchvision.ops import DeformConv2d


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 添加可学习的位置编码
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, embed_dim, 64, 64)
        )  # 初始大小为64x64，会自适应调整
        nn.init.normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x  # 存储输入用于残差连接

        # 添加位置编码
        pos_enc = F.interpolate(
            self.pos_encoding, size=(H, W), mode="bilinear", align_corners=False
        )
        x = x + pos_enc

        # 转换形状并进行自注意力处理
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # Reshape to (B, H*W, C)

        # 应用norm后再通过attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)

        # 第一个残差连接
        x_flat = x_flat + attn_out

        # 转换回原始形状
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)

        # 第二个残差连接（全局）
        return x + shortcut


class UNet(nn.Module):
    """
    改进的 U-Net 模块，添加跳跃连接，并使用 UpsamplingBilinear2d + Conv2d 进行上采样。
    同时移除了最大池化，改为 stride=2 的 Conv2d 下采样。
    """

    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.ModuleList()

        # 构建编码器部分
        for feature in features:
            self.encoder.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels,
                        feature,
                        kernel_size=7,
                        stride=1,
                        padding=3,
                        act_type="lrelu",
                        norm_type=None,
                    ),
                    ConvBlock(
                        feature,
                        feature,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        act_type="lrelu",
                        norm_type="bn",
                    ),
                )
            )
            self.pool.append(
                nn.Conv2d(
                    feature,
                    feature,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = feature

        # 构建解码器部分
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.Sequential(
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.Conv2d(
                        features[features.index(feature) + 1],
                        feature,
                        kernel_size=5,
                        padding=2,
                    ),
                    nn.LeakyReLU(negative_slope=0.2),
                    ConvBlock(
                        feature * 2,  # 跳跃连接的特征拼接
                        feature,
                        kernel_size=3,
                        padding=1,
                        act_type="lrelu",
                        norm_type=None,
                    ),
                )
            )

        # Bottleneck

        self.bottleneck = nn.Sequential(
            SelfAttention(features[-1], num_heads=4),
            ConvBlock(
                features[-1],
                features[-1] * 2,
                kernel_size=7,
                stride=1,
                padding=3,
                act_type="lrelu",
            ),
            ConvBlock(
                features[-1] * 2,
                features[-1],
                kernel_size=7,
                padding=3,
                act_type="lrelu",
                norm_type="bn",
            ),
        )

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], out_channels, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        # 编码器部分
        skip_connections = []
        for encoder_layer, pool_layer in zip(self.encoder, self.pool):
            x = encoder_layer(x)
            x = pool_layer(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1]  # 去掉最后一个跳跃连接
        # Bottleneck
        x = self.bottleneck(x)

        # 解码器部分
        skip_connections = skip_connections[::-1]
        for idx, decoder_layer in enumerate(self.decoder):
            x = decoder_layer[:2](x)  # 上采样
            skip_connection = skip_connections[idx]
            x = torch.cat((x, skip_connection), dim=1)  # concat 跳跃连接
            x = decoder_layer[2:](x)  # 解码器后续部分

        # 最终输出
        return self.final_conv(x)


class ISPNet(nn.Module):
    # 封装类，参数是形式不具体的batch，进来再解，返回的是具体的output和loss
    def __init__(self):
        super().__init__()
        self.unet = UNet(in_channels=20, out_channels=1)  # 替换为 U-Net
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.eps = 1e-5

    def forward(self, batch, stage="train"):
        # 返回output和loss供训练用

        if stage != "test":

            # calculate output
            result = ...  # self.net(...)

            # calculate loss
            pass

            loss_dict = {
                "loss": 0,
            }
            return result, loss_dict
        else:
            # test stage, only return output
            result = ...
            return result

    def ssim_loss(self, pred, target):
        return 1 - ssim(pred, target, nonnegative_ssim=True, data_range=1)


if __name__ == "__main__":
    Net = ISPNet()
    print(Net)
