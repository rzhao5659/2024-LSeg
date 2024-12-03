# Copied from github-repo: lang-seg

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt import DPT
# from .lseg_blocks import Interpolate, _make_encoder
import clip
import numpy as np
import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

class LSeg(BaseModel):
    def __init__(
        self,
        head,
        dpt,
        features=256,
        backbone="clip_vitb32_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        # hooks = {
        #     # "clip_vitl16_384": [5, 11, 17, 23],
        #     # "clipRN50x16_vitl16_384": [5, 11, 17, 23],
        #     "clip_vitb32_384": [2, 5, 8, 11],
        # }

        # # Instantiate backbone and reassemble blocks
        # self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
        #     backbone,
        #     features,
        #     groups=1,
        #     expand=False,
        #     exportable=False,
        #     hooks=hooks[backbone],
        #     use_readout=readout,
        # )
        self.clip_pretrained, _ = clip.load("ViT-B/32", jit=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self.out_c = 512
        self.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.output_conv = head
        
        self.dpt = dpt

        self.text = clip.tokenize(self.labels)    
        
    def forward(self, x, labelset=''):
        if labelset == '':
            text = self.text
        else:
            text = clip.tokenize(labelset)    
        
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)
        
        image_features = self.dpt(x)

        imshape = image_features.shape

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Reshape the image features
        batch, channels, height, width = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, channels)
        # print(image_features.shape)
        # print(text_features.shape)

        # Compute similarity
        logits = torch.matmul(image_features, text_features)
        logits_per_image = self.logit_scale * logits

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)
        # print(out.shape)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.head_block(out)
            out = self.head_block(out, False)

        out = self.output_conv(out)
        
        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        output_feature_dim = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels
        
        # self.out_c = 512
        # self.head1 = nn.Conv2d(output_feature_dim, self.out_c, kernel_size=1)

        num_classes_ADE20K = len(self.labels)
        dpt_head = nn.Sequential(
            nn.Conv2d(output_feature_dim, output_feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.1, False),
            nn.Conv2d(output_feature_dim, num_classes_ADE20K, kernel_size=1),
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )
        
        dpt = DPT(dpt_head, output_feature_dim)
        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, dpt=dpt, **kwargs)

        if path is not None:
            self.load(path)

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
