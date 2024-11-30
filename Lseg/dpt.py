import torch
from torch import nn
from Lseg.vit_wrapper import VisualTransformerWrapper
from Lseg.reassemble_blocks import ReassembleBlock
from Lseg.fusion_blocks import FeatureFusionBlock


class DPT(nn.Module):
    """
    Dense Prediction Transformer.

    Attributes:
        img_encoder (nn.Module): Default is ViT L/16 pretrained on ImageNet.
        head (nn.Module): Head of DPT to adapt for specific task.
        output_feature_dim: Final fusion block output feature dimension (before head).
        layers_to_hook (List[int] of size 4): Selected transformer layers from which we extract their outputs as features.
        reassemble_tokens_dim (List[int] of size 4): Output dimension (`Dhat`) of each reassemble block (ordered by layer)
    """

    def __init__(
        self,
        head: nn.Module,
        output_feature_dim: int,
        layers_to_hook=[4, 11, 17, 23],
        reassemble_tokens_dim=[256, 512, 1024, 1024],
    ):
        super().__init__()

        # DPT hyperparameters:
        self.layers_to_hook = layers_to_hook
        self.reassemble_token_dimensions = reassemble_tokens_dim
        self.output_feature_dim = output_feature_dim

        # DPT modules
        self.img_encoder = VisualTransformerWrapper(layers_to_hook)
        D_in = self.img_encoder.token_dimension
        self.reassemble_blocks = nn.ModuleList(
            [
                ReassembleBlock(s=4, D_in=D_in, D_out=reassemble_tokens_dim[0]),
                ReassembleBlock(s=8, D_in=D_in, D_out=reassemble_tokens_dim[1]),
                ReassembleBlock(s=16, D_in=D_in, D_out=reassemble_tokens_dim[2]),
                ReassembleBlock(s=32, D_in=D_in, D_out=reassemble_tokens_dim[3]),
            ]
        )
        self.fusion_blocks = nn.ModuleList(
            [
                FeatureFusionBlock(
                    input_token_dim=reassemble_tokens_dim[0], output_token_dim=output_feature_dim, use_bn=True
                ),
                FeatureFusionBlock(
                    input_token_dim=reassemble_tokens_dim[1], output_token_dim=reassemble_tokens_dim[0], use_bn=True
                ),
                FeatureFusionBlock(
                    input_token_dim=reassemble_tokens_dim[2], output_token_dim=reassemble_tokens_dim[1], use_bn=True
                ),
                FeatureFusionBlock(
                    input_token_dim=reassemble_tokens_dim[3], output_token_dim=reassemble_tokens_dim[2], use_bn=True
                ),
            ]
        )
        self.head = head

    def forward(self, X):
        # Call ViT L/16 and extract the layers outputs
        self.img_encoder(X)
        layers_outputs = self.img_encoder.layers_outputs

        # Pass them through Reassemble block to obtain image-like representations of shape (batch_size, H/s, W/s, D_out)
        layers_img_representations = [0] * 4
        for i in range(len(layers_outputs)):
            layers_img_representations[i] = self.reassemble_blocks[i](layers_outputs[i])

        # Fuse the image-like representations. Note that the iteration order is flipped.
        prev_fusion_output = torch.zeros_like(layers_img_representations[-1])
        for i in reversed(range(len(layers_outputs))):
            curr_fusion_output = self.fusion_blocks[i](prev_fusion_output, layers_img_representations[i])
            prev_fusion_output = curr_fusion_output

        # Pass through task-specific head.
        return self.head(curr_fusion_output)
