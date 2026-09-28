# ============================================================
# Image Captioning using ResNet + LSTM
# File: src/resnet.py
# Purpose: Custom ResNet-18-style image encoder
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# Basic Residual Block
# ============================================================

class BasicBlock(nn.Module):
    """
    Standard ResNet basic residual block.

    Architecture:
        Conv3x3 -> BatchNorm -> ReLU
        Conv3x3 -> BatchNorm
        + shortcut
        -> ReLU
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First 3x3 convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        # ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Shortcut connection
        # Needed when spatial size or number of channels changes.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual addition
        out = out + identity

        out = self.relu(out)

        return out


# ============================================================
# Custom ResNet-18 Encoder
# ============================================================

class ResNetEncoder(nn.Module):
    """
    ResNet-18-style encoder for image captioning.

    Input:
        [batch_size, 3, 224, 224]

    Output:
        [batch_size, 512]

    The final classification layer is intentionally removed
    because this network is used as an image feature extractor.
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        # ----------------------------------------------------
        # ResNet stages
        # ----------------------------------------------------

        self.layer1 = self._make_layer(
            in_channels=64,
            out_channels=64,
            blocks=2,
            stride=1
        )

        self.layer2 = self._make_layer(
            in_channels=64,
            out_channels=128,
            blocks=2,
            stride=2
        )

        self.layer3 = self._make_layer(
            in_channels=128,
            out_channels=256,
            blocks=2,
            stride=2
        )

        self.layer4 = self._make_layer(
            in_channels=256,
            out_channels=512,
            blocks=2,
            stride=2
        )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(
        self,
        in_channels,
        out_channels,
        blocks,
        stride
    ):
        """
        Construct one ResNet stage.
        """

        layers = []

        # First block may change spatial resolution
        layers.append(
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride
            )
        )

        # Remaining blocks preserve dimensions
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # Input:
        # [B, 3, 224, 224]

        x = self.stem(x)
        # [B, 64, 56, 56]

        x = self.layer1(x)
        # [B, 64, 56, 56]

        x = self.layer2(x)
        # [B, 128, 28, 28]

        x = self.layer3(x)
        # [B, 256, 14, 14]

        x = self.layer4(x)
        # [B, 512, 7, 7]

        x = self.avgpool(x)
        # [B, 512, 1, 1]

        x = torch.flatten(x, 1)
        # [B, 512]

        return x


# ============================================================
# Quick Architecture Test
# ============================================================

if __name__ == "__main__":

    # Create encoder
    encoder = ResNetEncoder()

    # Test input
    test_image = torch.randn(2, 3, 224, 224)

    # Forward pass
    features = encoder(test_image)

    print("Input shape:   ", test_image.shape)
    print("Output shape:  ", features.shape)

    # Count parameters
    total_params = sum(
        p.numel()
        for p in encoder.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    print("Total parameters:     ", total_params)
    print("Trainable parameters: ", trainable_params)