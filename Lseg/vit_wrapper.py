from torch import nn
from torchvision.models.vision_transformer import interpolate_embeddings
import torchvision
import functools
from typing import List


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
        self.vit_model = torchvision.models.vit_l_16(weights="IMAGENET1K_V1")
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