from torch import nn


class ResidualConvUnit(nn.Module):
    """
    Residual convolution module (Retains same dimension and spatial size).
    Batch normalization helps for segmentation.
    Copied from DPT repository'
    """

    def __init__(self, token_dim, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.use_bn,
        )
        self.conv2 = nn.Conv2d(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.use_bn,
        )
        self.relu = nn.ReLU()
        if self.use_bn is True:
            self.bn1 = nn.BatchNorm2d(token_dim)
            self.bn2 = nn.BatchNorm2d(token_dim)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, X):
        out = self.relu(X)
        out = self.conv1(out)
        if self.use_bn:
            out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        return self.skip_add.add(out, X)


class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block. Adapted from DPT repository
    Batch normalization helps for segmentation.
    """

    def __init__(self, input_token_dim: int, output_token_dim: int, use_bn: bool):
        super().__init__()
        self.res_conv1 = ResidualConvUnit(input_token_dim, use_bn)
        self.res_conv2 = ResidualConvUnit(input_token_dim, use_bn)
        self.project = nn.Conv2d(
            input_token_dim,
            output_token_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, *Xs):
        output = Xs[0] + self.res_conv1(Xs[1])
        output = self.res_conv2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        output = self.project(output)
        return output
