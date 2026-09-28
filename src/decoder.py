# ============================================================
# Image Captioning using ResNet + LSTM
# File: src/decoder.py
# Purpose: LSTM-based caption decoder
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# LSTM Decoder
# ============================================================

class LSTMDecoder(nn.Module):
    """
    LSTM decoder for image caption generation.

    Image features:
        [B, 512]

    Token embeddings:
        [B, T, 256]

    LSTM hidden size:
        512

    Vocabulary size:
        2541
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
        # Token embedding
        # ----------------------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # ----------------------------------------------------
        # Convert image features into initial LSTM states
        # ----------------------------------------------------

        self.feature_to_hidden = nn.Linear(
            feature_dim,
            hidden_dim
        )

        self.feature_to_cell = nn.Linear(
            feature_dim,
            hidden_dim
        )

        # ----------------------------------------------------
        # LSTM
        # ----------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Vocabulary prediction layer
        # ----------------------------------------------------

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def init_hidden_state(self, features):
        """
        Convert image features into the initial
        hidden state and cell state of the LSTM.

        Args:
            features:
                [B, 512]

        Returns:
            h0:
                [1, B, 512]

            c0:
                [1, B, 512]
        """

        h0 = self.feature_to_hidden(features)
        c0 = self.feature_to_cell(features)

        # LSTM expects:
        # [num_layers, batch_size, hidden_dim]

        h0 = h0.unsqueeze(0)
        c0 = c0.unsqueeze(0)

        return h0, c0

    def forward(self, features, captions):
        """
        Teacher-forced training forward pass.

        Args:
            features:
                Image features [B, 512]

            captions:
                Token IDs [B, T]

        Returns:
            logits:
                [B, T-1, vocab_size]
        """

        # ----------------------------------------------------
        # Remove <END> target token from decoder input.
        # ----------------------------------------------------

        inputs = captions[:, :-1]

        # ----------------------------------------------------
        # Convert token IDs into embeddings.
        # ----------------------------------------------------

        embeddings = self.embedding(inputs)
        # [B, T-1, 256]

        # ----------------------------------------------------
        # Initialize LSTM states from image features.
        # ----------------------------------------------------

        h0, c0 = self.init_hidden_state(features)

        # ----------------------------------------------------
        # LSTM forward pass.
        # ----------------------------------------------------

        outputs, _ = self.lstm(
            embeddings,
            (h0, c0)
        )

        # ----------------------------------------------------
        # Convert hidden states into vocabulary logits.
        # ----------------------------------------------------

        logits = self.fc(outputs)

        return logits


# ============================================================
# Quick Decoder Test
# ============================================================

if __name__ == "__main__":

    # Example dimensions from the trained project
    batch_size = 2
    sequence_length = 20
    vocab_size = 2541

    # Create decoder
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        feature_dim=512,
        embedding_dim=256,
        hidden_dim=512
    )

    # Fake image features
    features = torch.randn(
        batch_size,
        512
    )

    # Fake tokenized captions
    captions = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length)
    )

    # Forward pass
    logits = decoder(
        features,
        captions
    )

    print("Features shape: ", features.shape)
    print("Captions shape: ", captions.shape)
    print("Logits shape:   ", logits.shape)

    # Parameter count
    total_params = sum(
        p.numel()
        for p in decoder.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in decoder.parameters()
        if p.requires_grad
    )

    print("Total parameters:     ", total_params)
    print("Trainable parameters: ", trainable_params)