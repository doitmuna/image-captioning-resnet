# ============================================================
# Image Captioning using ResNet + LSTM
# File: src/model.py
# Purpose: Complete image captioning model
# ============================================================

import torch
import torch.nn as nn

from resnet import ResNetEncoder
from decoder import LSTMDecoder


# ============================================================
# Complete Image Captioning Model
# ============================================================

class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model.

    Pipeline:

        Image
          ↓
        ResNet Encoder
          ↓
        512-dimensional image features
          ↓
        LSTM Decoder
          ↓
        Vocabulary logits

    The architecture matches the model used during training.
    """

    def __init__(
        self,
        vocab_size=2541,
        feature_dim=512,
        embedding_dim=256,
        hidden_dim=512
    ):
        super().__init__()

        # ----------------------------------------------------
        # Image encoder
        # ----------------------------------------------------

        self.encoder = ResNetEncoder()

        # ----------------------------------------------------
        # Caption decoder
        # ----------------------------------------------------

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, images, captions):
        """
        Training forward pass.

        Args:
            images:
                [B, 3, 224, 224]

            captions:
                [B, T]

        Returns:
            logits:
                [B, T-1, vocab_size]
        """

        # Extract image features
        features = self.encoder(images)
        # [B, 512]

        # Generate caption logits
        logits = self.decoder(
            features,
            captions
        )

        return logits

    def encode_image(self, images):
        """
        Extract image features.

        Args:
            images:
                [B, 3, 224, 224]

        Returns:
            features:
                [B, 512]
        """

        return self.encoder(images)


# ============================================================
# Parameter Count Utility
# ============================================================

def count_parameters(model):
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Quick Model Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Configuration from the trained project
    # --------------------------------------------------------

    vocab_size = 2541
    feature_dim = 512
    embedding_dim = 256
    hidden_dim = 512

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    # --------------------------------------------------------
    # Fake test data
    # --------------------------------------------------------

    images = torch.randn(
        2,
        3,
        224,
        224
    )

    captions = torch.randint(
        0,
        vocab_size,
        (2, 20)
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits = model(
        images,
        captions
    )

    # --------------------------------------------------------
    # Print shapes
    # --------------------------------------------------------

    print("Images shape:   ", images.shape)
    print("Captions shape: ", captions.shape)
    print("Logits shape:   ", logits.shape)

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total, trainable = count_parameters(model)

    print("Total parameters:     ", total)
    print("Trainable parameters: ", trainable)