import torch
import torch.nn as nn
from .dpt import DPT
import clip


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

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
        self.temperature = nn.Parameter(torch.tensor(0.07))
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
        image_features = image_features.permute(0, 2, 3, 1).view(-1, channels)

        # Compute similarity
        logits = torch.matmul(image_features, torch.transpose(text_features, 0, 1))
        logits_per_image = torch.exp(self.temperature) * logits
        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        # Just upscale the prediction to original input resolution
        out = self.head(out)
        return out
