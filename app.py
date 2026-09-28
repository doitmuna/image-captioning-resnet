import os
import sys

import streamlit as st
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from inference import load_model, generate_caption_with_details
from vocabulary import Vocabulary

MODEL_PATH = os.path.join(ROOT, "models", "final_model.pth")
VOCAB_PATH = os.path.join(ROOT, "config", "vocabulary.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    page_icon="🖼️",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: #071A12;
        color: #F3F7F4;
    }

    [data-testid="stHeader"] {
        background: #071A12;
    }

    [data-testid="stSidebar"] {
        background: #0B2419;
    }

    .main {
        background: #071A12;
    }

    .block-container {
        max-width: 1150px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        color: #F4F8F5;
    }

    .hero p {
        font-size: 1.1rem;
        color: #A8BDB0;
        margin-bottom: 0;
    }

    .accent {
        color: #6FD08C;
    }

    .upload-card {
        background: #0D281C;
        border: 1px solid #1E4631;
        border-radius: 18px;
        padding: 1.4rem;
        margin-top: 1.5rem;
    }

    .caption-card {
        background: #0D281C;
        border: 1px solid #2B6243;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1rem;
    }

    .caption-label {
        color: #7CCF93;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .caption-text {
        color: #F4F8F5;
        font-size: 1.45rem;
        line-height: 1.6;
        font-weight: 600;
    }

    .info-card {
        background: #0A2117;
        border: 1px solid #1C3E2C;
        border-radius: 14px;
        padding: 1rem;
        color: #B4C6BB;
        font-size: 0.92rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: #68C483;
        color: #06130C;
        font-weight: 800;
        font-size: 1rem;
        padding: 0.7rem 1rem;
    }

    .stButton > button:hover {
        background: #7AD394;
        color: #06130C;
    }

    .stFileUploader {
        background: transparent;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0A2117;
        border: 1px dashed #376B4B;
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #B9CABF;
    }

    .stExpander {
        background: #0A2117;
        border: 1px solid #1C3E2C;
        border-radius: 12px;
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
        <h1>Neural <span class="accent">Image Captioning</span></h1>
        <p>Generate natural-language descriptions from images using a custom ResNet + LSTM model.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(
        [1.15, 1],
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
            <div class="info-card">
            <strong>Model</strong><br>
            Custom ResNet-18-style encoder + LSTM decoder<br><br>
            <strong>Decoding</strong><br>
            Greedy autoregressive generation<br><br>
            <strong>Vocabulary</strong><br>
            2,541 tokens
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            "Generate Caption",
            type="primary"
        ):

            with st.spinner("Generating caption..."):

                result = generate_caption_with_details(
                    model=model,
                    image=image,
                    vocabulary=vocabulary,
                    device=device
                )

            st.markdown(
                f"""
                <div class="caption-card">
                    <div class="caption-label">Generated Caption</div>
                    <div class="caption-text">
                        {result["caption"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            with st.expander("Token-level confidence"):

                for token, probability in zip(
                    result["tokens"],
                    result["probabilities"]
                ):

                    st.write(
                        f"**{token}** — {probability:.2%}"
                    )

            st.caption(
                f"Inference device: {device}"
            )

else:

    st.markdown(
        """
        <div class="info-card" style="text-align:center; margin-top:1.5rem;">
            Upload an image above to generate a caption.
        </div>
        """,
        unsafe_allow_html=True
    )