import os
import sys

import torch
from torchvision import transforms
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import ImageCaptioningModel
from vocabulary import Vocabulary


def get_image_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def load_model(model_path, vocab_size, device):
    model = ImageCaptioningModel(
        vocab_size=vocab_size,
        feature_dim=512,
        embedding_dim=256,
        hidden_dim=512
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def generate_caption(
    model,
    image,
    vocabulary,
    device,
    max_length=30
):
    transform = get_image_transform()

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    features = model.encoder(image_tensor)

    hidden, cell = model.decoder.init_hidden_state(
        features
    )

    current_token = torch.tensor(
        [vocabulary.start_id],
        dtype=torch.long,
        device=device
    )

    generated_ids = []

    for _ in range(max_length):
        embedding = model.decoder.embedding(
            current_token
        )

        embedding = embedding.unsqueeze(1)

        output, (hidden, cell) = model.decoder.lstm(
            embedding,
            (hidden, cell)
        )

        logits = model.decoder.fc(
            output.squeeze(1)
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.argmax(
            probabilities,
            dim=-1
        )

        token_id = next_token.item()

        if token_id == vocabulary.end_id:
            break

        if token_id != vocabulary.pad_id:
            generated_ids.append(token_id)

        current_token = next_token

    return vocabulary.decode_sentence(
        generated_ids
    )


@torch.no_grad()
def generate_caption_with_details(
    model,
    image,
    vocabulary,
    device,
    max_length=30
):
    transform = get_image_transform()

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    features = model.encoder(image_tensor)

    hidden, cell = model.decoder.init_hidden_state(
        features
    )

    current_token = torch.tensor(
        [vocabulary.start_id],
        dtype=torch.long,
        device=device
    )

    generated_ids = []
    generated_tokens = []
    token_probabilities = []

    for _ in range(max_length):
        embedding = model.decoder.embedding(
            current_token
        )

        embedding = embedding.unsqueeze(1)

        output, (hidden, cell) = model.decoder.lstm(
            embedding,
            (hidden, cell)
        )

        logits = model.decoder.fc(
            output.squeeze(1)
        )

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.argmax(
            probabilities,
            dim=-1
        )

        token_id = next_token.item()

        probability = probabilities[
            0,
            token_id
        ].item()

        if token_id == vocabulary.end_id:
            break

        if token_id != vocabulary.pad_id:
            token = vocabulary.index_to_word(
                token_id
            )

            generated_ids.append(token_id)
            generated_tokens.append(token)
            token_probabilities.append(probability)

        current_token = next_token

    return {
        "caption": " ".join(generated_tokens),
        "tokens": generated_tokens,
        "token_ids": generated_ids,
        "probabilities": token_probabilities
    }


def caption_image(
    image_path,
    model,
    vocabulary,
    device
):
    image = Image.open(
        image_path
    ).convert("RGB")

    return generate_caption(
        model=model,
        image=image,
        vocabulary=vocabulary,
        device=device
    )


if __name__ == "__main__":
    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        "final_model.pth"
    )

    vocab_path = os.path.join(
        PROJECT_ROOT,
        "config",
        "vocabulary.json"
    )

    image_path = os.path.join(
        PROJECT_ROOT,
        "test_images",
        "test.png"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocabulary = Vocabulary(
        vocab_path
    )

    model = load_model(
        model_path=model_path,
        vocab_size=vocabulary.size,
        device=device
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    caption = generate_caption(
        model=model,
        image=image,
        vocabulary=vocabulary,
        device=device
    )

    print("Image:", image_path)
    print("Device:", device)
    print("Caption:", caption)