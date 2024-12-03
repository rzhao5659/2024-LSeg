from torch import nn
# from torchvision.models.vision_transformer import interpolate_embeddings
import torchvision
import torchvision.models as models
import functools
from typing import List

import math

from collections import OrderedDict
import torch
import torch.nn as nn


class VisualTransformerWrapper(nn.Module):
    """
    This will provide additional functionalities to the VisualTransformer from torchvision.
    In particular, forward hooks to store specific layers outputs and
    somewhat arbitrary input size (H = W has to match) via position embedding interpolation.

    Attributes:
        vit_model (nn.Module): ViT L/16 model pretrained on ImageNet.
        token_dimension (int): Input token dimension (`D` in DPT's paper). Default is 1024.
        patch_size (int): Patch's size (`p`) of input image over which a token embedding is extracted. Default is 16.
        layers_to_hook (List[int] of size 4): Selected transformer layers from which we extract their outputs as features.
        layers_outputs (List[int] of size 4): Stores selected transformer layers outputs.
    """

    def __init__(self, layers_to_hook: List[int]):
        super().__init__()
        # self.vit_model = torchvision.models.vit_l_16(weights="IMAGENET1K_V1")
        self.vit_model = torchvision.models.resnet101(pretrained=True)
        self.token_dimension = 1024
        self.patch_size = 16
        self.layers_to_hook = layers_to_hook
        self.layers_outputs = [0] * len(layers_to_hook)

        # Register callbacks on forward() of these specified transformer layers.
        # These callback will simply store their output values.
        for i, layer_idx in enumerate(self.layers_to_hook):
            layer = getattr(self.vit_model.encoder.layers, f"encoder_layer_{layer_idx}")
            layer.register_forward_hook(functools.partial(self._store_layer_output, storage_idx=i))

    def forward(self, X):
        # Check if we need to adjust position embeddings:
        # ViT model are pretrained with specific input image size.
        # If given a different image size, interpolate the learned position embeddings.
        new_img_size = X.shape[-1]
        assert new_img_size % self.patch_size == 0, "new image size not divisible by 16."
        assert X.shape[-1] == X.shape[-2], "Height and width must be same size"
        if self.vit_model.image_size != new_img_size:
            self.vit_model.image_size = new_img_size
            new_model_state = interpolate_embeddings(
                new_img_size, patch_size=16, model_state=self.vit_model.state_dict()
            )
            self.vit_model.encoder.pos_embedding.data = new_model_state["encoder.pos_embedding"]

        return self.vit_model(X)

    def _store_layer_output(self, module, input, output, storage_idx) -> None:
        """
        Callback function hooked on specific layer's forward() call.
        This will simply store that layer's output.
        """
        self.layers_outputs[storage_idx] = output

# Copied from the source of torchvision.models.vision_transformer.py
def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state