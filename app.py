import os
import sys

import streamlit as st
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
EXAMPLES = os.path.join(ROOT, "examples")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference import load_model, generate_caption_with_details
from vocabulary import Vocabulary

MODEL_PATH = os.path.join(ROOT, "models", "final_model.pth")
VOCAB_PATH = os.path.join(ROOT, "config", "vocabulary.json")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@st.cache_resource
def load_resources():
    vocabulary = Vocabulary(VOCAB_PATH)
    model = load_model(
        model_path=MODEL_PATH,
        vocab_size=vocabulary.size,
        device=device
    )
    return model, vocabulary


model, vocabulary = load_resources()

st.set_page_config(
    page_title="Neural Image Captioning",
    page_icon="◈",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #071A12;
        color: #EAF3ED;
    }

    [data-testid="stHeader"] {
        background: #071A12;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, div, span, label, button {
        font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    }

    .hero {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        color: #F3F8F5;
        margin-bottom: 0.6rem;
    }

    .hero p {
        color: #9FB6A8;
        font-size: 1rem;
        line-height: 1.7;
    }

    .green {
        color: #73D391;
    }

    .card {
        background: #0B2419;
        border: 1px solid #1E4631;
        border-radius: 18px;
        padding: 1.2rem;
    }

    .caption-card {
        background: #0A2117;
        border: 1px solid #2D6645;
        border-radius: 16px;
        padding: 1.25rem;
        margin-top: 1rem;
    }

    .caption-label {
        color: #73D391;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .caption-text {
        color: #F2F7F4;
        font-size: 1.15rem;
        line-height: 1.7;
        font-weight: 700;
    }

    .info {
        background: #0A2117;
        border: 1px solid #1B3D2B;
        border-radius: 14px;
        padding: 1rem;
        color: #AFC2B5;
        line-height: 1.7;
        font-size: 0.82rem;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #EAF3ED;
        margin: 2rem 0 1rem 0;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #68C483;
        background: #68C483;
        color: #06130C;
        font-weight: 900;
        padding: 0.7rem 1rem;
    }

    .stButton > button:hover {
        background: #7DDB98;
        border-color: #7DDB98;
        color: #06130C;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0A2117;
        border: 1px dashed #3E7353;
        border-radius: 15px;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        <h1>Neural <span class="green">Image Captioning</span></h1>
        <p>
            From-scratch image captioning with a custom ResNet-based encoder
            and LSTM decoder.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if os.path.isdir(EXAMPLES):
    example_files = sorted(
        [
            f for f in os.listdir(EXAMPLES)
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp",
                    ".tif",
                    ".tiff",
                    ".gif"
                )
            )
        ]
    )

    if example_files:
        st.markdown(
            '<div class="section-title">Curated examples</div>',
            unsafe_allow_html=True
        )

        columns = st.columns(min(len(example_files), 4))

        for index, filename in enumerate(example_files):
            with columns[index % len(columns)]:
                image_path = os.path.join(
                    EXAMPLES,
                    filename
                )

                image = Image.open(
                    image_path
                ).convert("RGB")

                st.image(
                    image,
                    use_container_width=True
                )

                if st.button(
                    "Generate",
                    key=f"example_{index}"
                ):
                    with st.spinner("Generating..."):
                        result = generate_caption_with_details(
                            model=model,
                            image=image,
                            vocabulary=vocabulary,
                            device=device
                        )

                    st.markdown(
                        f"""
                        <div class="caption-card">
                            <div class="caption-label">
                                Generated caption
                            </div>
                            <div class="caption-text">
                                {result["caption"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

st.markdown(
    '<div class="section-title">Upload an image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "gif"
    ],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    left, right = st.columns(
        [1.2, 1],
        gap="large"
    )

    with left:
        st.image(
            image,
            use_container_width=True
        )

    with right:
        st.markdown(
            """
            <div class="info">
                <strong>ENCODER</strong><br>
                Custom ResNet-18-style encoder
                <br><br>
                <strong>DECODER</strong><br>
                LSTM caption decoder
                <br><br>
                <strong>VOCABULARY</strong><br>
                2,541 tokens
                <br><br>
                <strong>DECODING</strong><br>
                Greedy autoregressive generation
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "Generate Caption",
            type="primary"
        ):
            with st.spinner("Running image captioning..."):
                result = generate_caption_with_details(
                    model=model,
                    image=image,
                    vocabulary=vocabulary,
                    device=device
                )

            st.markdown(
                f"""
                <div class="caption-card">
                    <div class="caption-label">
                        Generated caption
                    </div>
                    <div class="caption-text">
                        {result["caption"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Token-level probabilities"):
                for token, probability in zip(
                    result["tokens"],
                    result["probabilities"]
                ):
                    st.write(
                        f"{token} — {probability:.2%}"
                    )

            st.caption(
                f"Inference device: {device}"
            )

st.markdown(
    """
    <div style="
        text-align:center;
        color:#698273;
        font-size:0.72rem;
        margin-top:3rem;
    ">
        ResNet Encoder · LSTM Decoder · Flickr8k
    </div>
    """,
    unsafe_allow_html=True
)