import torch
from torch import nn
import math


class ReadProjectBlock(nn.Module):
    """
    This handles how to "add" the readout/cls token to the Np tokens from the image, using the projection approach:
    1) Concatenates the cls token to all Np tokens (dimension 2D)
    2) Project each token back to their original dimension D using a linear layer with GELU activation.
    Shapes: (batch_size, number_tokens=Np + 1, token_dim=D) -> (batch_size, Np, D)
    """

    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim
        self.project_layer = nn.Sequential(
            nn.Linear(in_features=2 * token_dim, out_features=token_dim),
            nn.GELU(),
        )

    def forward(self, X):
        # Assumption: The class token is the first token, following implementation of torchvision.models.vit_l_16
        assert X.dim() == 3, f"Expected shape (batch_size, number_tokens = Np + 1,  token_dim = D), got {X.shape}"
        batch_size, num_tokens, token_dim = X.shape
        assert token_dim == self.token_dim, f"Expected token dimension {self.token_dim}, got {token_dim}"

        # Extract the class tokens of the batch
        batch_cls_tokens = X[:, 0]

        # Concatenate these into the other tokens, yielding Np tokens with new dimension 2D:
        # Unsqueeze (batch_size, D) -> (batch_size, 1, D), then expand class tokens -> (batch_size, Np, D)
        # Concatenate cls token with the other Np tokens (batch_size, Np, D) -> (batch_size, Np, 2D)
        batch_cls_tokens = batch_cls_tokens.unsqueeze(1).expand_as(X[:, 1:])
        X = torch.cat([X[:, 1:], batch_cls_tokens], dim=-1)

        # Project Np tokens of dimension 2D back to D.
        X = self.project_layer(X.view(-1, 2 * token_dim))
        X = X.view((batch_size, num_tokens - 1, token_dim))
        return X


class ConcatenateBlock(nn.Module):
    """
    Reshape the sequence of Np tokens (batch_size, Np, D) into an image-like representation (batch_size, H/p, W/p, D).
    The Np input tokens corresponds one-to-one to the Np patches of image, hence we can put them back in image shape.

    Determines H, W during runtime assuming H = W.

    Terminology:
    H, W are input image resolutions. p is patch size, D is input token dimension.
    Np is number of tokens and it satisfies Np = H/p * W/p.
    """

    def __init__(self, patch_size=16):
        super().__init__()
        self.p = patch_size
        self.last_num_tokens = None

    def forward(self, X):
        batch_size, num_tokens, token_dim = X.shape
        if self.last_num_tokens != num_tokens:
            self.last_num_tokens = num_tokens
        H_patches = int(math.sqrt(self.last_num_tokens))
        W_patches = H_patches
        return X.view(batch_size, H_patches, W_patches, token_dim)


class ResampleBlock(nn.Module):
    """Process (batch_size, H/p, W/p, D) image to (batch_size, D_out, H/s, W/s) image.

    It does the following process:
    1) Reshape to pytorch convention (batch_size, H/p, W/p, D) -> (batch_size, D, H/p, W/p)
    1) Use 1x1 convolution to project (D, H/p, W/p) to (D_out, H/p, W/p)
    2) Downsample or upsample (depending on s) the image

    s is the output image size ratio (relative to input image).
    It must be one of these values: (4, 8, 16, 32), which are the only values needed for DPT.
    The specific ConvLayer that downsamples or upsamples for that s are hardcoded.
    """

    admissible_s = [4, 8, 16, 32]

    def __init__(self, s, input_token_dim, output_token_dim, patch_size=16):
        super().__init__()
        self.p = patch_size
        self.s = s
        self.D_in = input_token_dim
        self.D_out = output_token_dim

        # Verify s
        assert self.s in ResampleBlock.admissible_s, f"Output size ratio {self.s} is unsupported."

        # Define 1x1 conv layer
        self.pointwise_conv_layer = nn.Conv2d(in_channels=self.D_in, out_channels=self.D_out, kernel_size=1)

        # Select one of these hardcoded spatial resampling layer based on s.
        # These layers parameters follow DPT implementation.
        if self.s == 4:
            # Upsample by 4
            self.spatial_resample_layer = nn.ConvTranspose2d(
                in_channels=self.D_out,
                out_channels=self.D_out,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif self.s == 8:
            # Upsample by 2
            self.spatial_resample_layer = nn.ConvTranspose2d(
                in_channels=self.D_out,
                out_channels=self.D_out,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif self.s == 16:
            # Use identity as it remains with same spatial resolution.
            self.spatial_resample_layer = nn.Identity()
        elif self.s == 32:
            # Downsample by 2
            self.spatial_resample_layer = nn.Conv2d(
                in_channels=self.D_out,
                out_channels=self.D_out,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        else:
            assert False, "Shouldn't arrive here."

    def forward(self, X):
        # Permute (batch_size, H/p, W/p, D) -> (batch_size, D, H/p, W/p) before applying conv layer
        X = torch.permute(X, (0, 3, 1, 2))
        # Apply 1x1 conv layer: (batch_size, D, H/p, W/p) -> (batch_size, D_out, H/p, W/p)
        X = self.pointwise_conv_layer(X)
        # Apply spatial resampling layer (batch_size, D_out, H/p, W/p) -> (batch_size, D_out, H/s, W/s)
        X = self.spatial_resample_layer(X)
        return X


class ReassembleBlock(nn.Module):
    def __init__(self, s, D_in, D_out):
        """
        Reassemble is a three-staged operation to obtain an image-like representation from an input token:
        img_like_representation = (Resample o Concatenate o Read)(token)

        Attributes:
            s = output size ratio of the representation wrt the input image.
            D_out = output token's dimension
            D_in = input token's dimension
        """
        super().__init__()
        self.s = s
        self.D_in = D_in
        self.D_out = D_out
        self.reassemble = nn.Sequential(
            ReadProjectBlock(token_dim=self.D_in),
            ConcatenateBlock(patch_size=16),
            ResampleBlock(s=self.s, input_token_dim=self.D_in, output_token_dim=self.D_out, patch_size=16),
        )

    def forward(self, X):
        return self.reassemble(X)
