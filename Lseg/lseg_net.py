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
    def __init__(self, kernel_size=3, stride=1, padding=1, activation="relu"):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation="relu"):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
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


class LSegNet(BaseModel):
    """Network for semantic segmentation."""

    def __init__(self, labels, path=None, **kwargs):
        super().__init__()
        multimodal_embedding_dim = kwargs["features"] if "features" in kwargs else 512
        kwargs["use_bn"] = True

        # Segmentation text labels
        self.labels = labels

        # Text encoder
        self.clip_pretrained, _ = clip.load("ViT-B/32", jit=False)
        self.clip_pretrained.requires_grad_(False)
        self.clip_pretrained = self.clip_pretrained.to(device="cuda")

        # Image encoder
        self.dpt = DPT(head=nn.Identity(), output_feature_dim=multimodal_embedding_dim)

        # LSeg parameters
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self.text_tokens = clip.tokenize(self.labels).to(device="cuda")
        self.head = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

        if path is not None:
            self.load(path)

    def forward(self, x, labelset=""):
        if labelset == "":
            text = self.text_tokens
        else:
            text = clip.tokenize(labelset)

        text_features = self.clip_pretrained.encode_text(text)
        image_features = self.dpt(x)

        imshape = image_features.shape

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Reshape the image features
        batch, channels, height, width = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, channels)

        # Compute similarity
        logits = torch.matmul(image_features, text_features.T)
        logits_per_image = self.logit_scale * logits

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        # Just upscale the prediction to original input resolution
        out = self.head(out)

        return out


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
