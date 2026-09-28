## Live Demo

[🚀 Try the Streamlit Demo](https://image-captioning-resnet-odolvra5bkmyxzej2wffes.streamlit.app)
# Neural Image Captioning

A from-scratch image captioning system using a custom ResNet-18-style CNN encoder and an LSTM-based caption decoder trained on Flickr8k.

> **Project status:** Baseline model completed, evaluated, and packaged for local inference and Streamlit deployment.

---

## Overview

Image captioning is the task of generating a natural-language description of an image.

This project implements the complete image captioning pipeline without using pretrained model weights:

```text
Image
  ↓
Custom ResNet Encoder
  ↓
512-dimensional Image Feature
  ↓
Feature Projection
  ↓
LSTM Decoder
  ↓
Vocabulary Probabilities
  ↓
Generated Caption
```

The project was built to understand the complete system from the model architecture to training, evaluation, inference, and deployment rather than relying on a pretrained image-captioning pipeline.

---

## Project Goals

The main goals of the project were to:

- Implement the CNN image encoder from scratch
- Implement the LSTM caption decoder from scratch
- Understand residual learning and skip connections
- Build the vocabulary and caption-tokenization pipeline
- Train the system using teacher forcing
- Evaluate generated captions quantitatively
- Analyze qualitative failure cases
- Build a reusable inference pipeline
- Package the trained model for deployment
- Build a Streamlit application

---

## Architecture

### Complete Pipeline

```text
                  IMAGE
                    │
                    ▼
        ┌──────────────────────┐
        │ Custom ResNet Encoder│
        └──────────────────────┘
                    │
                    ▼
          512-D Image Feature
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
 Feature → Hidden       Feature → Cell
          │                   │
          └─────────┬─────────┘
                    ▼
             Initial LSTM State
                    │
                    ▼
              Word Embedding
                    │
                    ▼
                  LSTM
                    │
                    ▼
             Linear Projection
                    │
                    ▼
          Vocabulary Probabilities
                    │
                    ▼
            Generated Caption
```

### Encoder

The image encoder is a custom ResNet-18-style architecture implemented without pretrained weights.

```text
Input
224 × 224 × 3
      ↓
Stem
      ↓
64 × 56 × 56
      ↓
Residual Layer 1
64 × 56 × 56
      ↓
Residual Layer 2
128 × 28 × 28
      ↓
Residual Layer 3
256 × 14 × 14
      ↓
Residual Layer 4
512 × 7 × 7
      ↓
Adaptive Average Pooling
      ↓
512-dimensional feature vector
```

The encoder contains approximately **11.18 million parameters**.

### Decoder

The decoder uses:

| Component | Dimension |
|---|---:|
| Image feature | 512 |
| Word embedding | 256 |
| LSTM hidden state | 512 |
| Vocabulary | 2,541 |

The image feature is projected into the initial hidden and cell states of the LSTM.

---

## Why ResNet?

A standard convolutional network can learn increasingly abstract visual representations as depth increases.

ResNet introduces residual connections that allow a block to learn a residual transformation:

```text
Input
  │
  ├──────────────┐
  │              │
  ▼              │
Convolution      │
  ↓              │
BatchNorm        │
  ↓              │
ReLU             │
  ↓              │
Convolution      │
  ↓              │
BatchNorm        │
  │              │
  └────── + ─────┘
         │
         ▼
       ReLU
```

The skip connection allows the input to bypass the convolutional transformation.

In this project, the ResNet is used as a visual feature extractor rather than a classification network.

---

## Why LSTM?

An image alone does not produce a sentence.

The decoder must generate a sequence:

```text
<START>
    ↓
a
    ↓
boy
    ↓
is
    ↓
running
    ↓
through
    ↓
a
    ↓
field
    ↓
<END>
```

The LSTM maintains a hidden state and cell state that carry information through the sequence.

At each timestep:

```text
Previous Token
      +
Previous Hidden State
      +
Previous Cell State
      ↓
     LSTM
      ↓
Next Token Probability
```

---

## Dataset

The project uses the **Flickr8k** image-captioning dataset.

### Dataset Split

| Split | Images | Caption Samples |
|---|---:|---:|
| Training | 6,000 | 30,000 |
| Validation | 1,000 | 5,000 |
| Test | 1,000 | 5,000 |

Each image has five human-written reference captions.

The splits are performed at the image level so that images from the test set do not appear in the training set.

---

## Caption Preprocessing

Captions are tokenized and surrounded by special tokens:

```text
<START> caption words <END>
```

The vocabulary uses:

```text
<PAD>   = 0
<START> = 1
<END>   = 2
<UNK>   = 3
```

A minimum frequency threshold of **5** was used when constructing the vocabulary.

Final vocabulary size:

```text
2,541 tokens
```

Words that do not occur in the final vocabulary are mapped to `<UNK>`.

---

## Image Preprocessing

Input images are resized to:

```text
224 × 224
```

Training preprocessing includes random horizontal flipping.

Validation and test preprocessing is deterministic.

Images are normalized using:

```text
mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]
```

---

## Training

The baseline model was trained from scratch.

### Training Configuration

| Parameter | Value |
|---|---:|
| Epochs | 15 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-5 |
| Loss | Cross-Entropy |
| Gradient Clipping | 5.0 |

### Teacher Forcing

Teacher forcing is used during training.

During training, the decoder receives the correct previous token and learns to predict the next token.

For example:

```text
Input:

<START> a boy is

Target:

a boy is running
```

The objective is next-token prediction.

### Padding

Captions have different lengths, so sequences are padded to a common length.

Padding tokens are ignored by the loss function:

```python
CrossEntropyLoss(ignore_index=<PAD>)
```

This prevents artificial learning from padded positions.

---

## Training Results

The baseline training loss decreased from:

```text
Epoch 1:
4.4402
```

to:

```text
Epoch 15:
2.4043
```

Validation loss decreased from:

```text
Epoch 1:
3.8613
```

to:

```text
Epoch 15:
2.9881
```

### Best Validation Loss

```text
2.9881
```

The training and validation curves are available in:

```text
results/training_validation_loss.png
```

---

## Evaluation

The trained model was evaluated on:

```text
1,000 unique test images
```

The generated captions were compared against the original human-written reference captions.

### Metrics

The project reports:

- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4
- METEOR
- ROUGE-L

### Quantitative Results

| Metric | Score |
|---|---:|
| BLEU-1 | 0.5136 |
| BLEU-2 | 0.3213 |
| BLEU-3 | 0.1955 |
| BLEU-4 | 0.1223 |
| METEOR | 0.3151 |
| ROUGE-L | 0.3833 |

These numbers represent the measured performance of this implementation on the specified Flickr8k test split.

They are **not presented as state-of-the-art results**.

---

## Qualitative Analysis

Quantitative metrics alone do not fully describe captioning quality.

The project also includes qualitative inspection of generated captions.

The model generally learns broad scene-level information, but several recurring failure modes were observed.

### Observed Failure Modes

- Wrong object
- Wrong action
- Wrong object count
- Wrong color or attribute
- Confusion between visually similar objects
- Difficulty representing interactions between multiple people
- Reduced accuracy on images outside the training distribution

### Example 1 — Bicycle Scene

Reference descriptions describe two boys riding a bicycle.

Generated caption:

```text
a man in a blue shirt is standing on a <UNK> road
```

### Example 2 — ATV Scene

Reference descriptions describe a man driving an ATV.

Generated caption:

```text
a man in a red shirt is riding a bike through the woods
```

### Example 3 — Football Scene

Reference descriptions describe two football players interacting during a tackle.

Generated caption:

```text
a football player is running on the field
```

These examples show that the model can capture the general scene while still failing to identify fine-grained objects, attributes, and interactions.

---

## Out-of-Distribution Behavior

The model was also tested on images outside the Flickr8k test distribution.

For example, a modern photograph of a child running through a field produced:

```text
a little girl in a pink shirt is running through a field
```

while the image actually showed a boy wearing a yellow shirt.

This demonstrates an important limitation of the current baseline:

> The model can produce a plausible sentence that matches the general scene while getting specific visual attributes wrong.

This is one reason why qualitative analysis is included alongside automatic captioning metrics.

---

## Inference

The inference pipeline uses greedy decoding.

```text
Input Image
     ↓
Resize + Normalize
     ↓
ResNet Encoder
     ↓
Image Feature
     ↓
Initialize LSTM
     ↓
<START>
     ↓
Predict next token
     ↓
Feed token back into LSTM
     ↓
Repeat
     ↓
<END>
     ↓
Generated Caption
```

At each step, the token with the highest predicted probability is selected.

---

## Token-Level Confidence

The inference pipeline can also expose the probability assigned to each generated token.

Example:

```text
two       69.50%
black     75.98%
dogs      87.25%
are       25.77%
running   54.36%
```

This makes the inference pipeline useful not only as a demo but also as a simple way to inspect model decisions.

---

## Streamlit Application

The project includes an interactive Streamlit interface.

The application provides:

- Image upload
- Image preview
- Caption generation
- Token-level probability display
- Model/device information

Supported image formats include:

```text
JPG
JPEG
PNG
WEBP
BMP
TIF
TIFF
GIF
```

The deployment application loads the trained checkpoint and vocabulary directly from the project repository.

---

## Project Structure

```text
image-captioning-resnet/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── vocabulary.json
│   └── inference_config.json
│
├── models/
│   └── final_model.pth
│
├── results/
│   ├── evaluation_metrics.json
│   ├── failure_cases.json
│   ├── generated_captions.json
│   └── training_validation_loss.png
│
└── src/
    ├── decoder.py
    ├── inference.py
    ├── model.py
    ├── resnet.py
    └── vocabulary.py
```

---

## Local Setup

Clone the repository:

```bash
git clone https://github.com/doitmuna/image-captioning-resnet.git
cd image-captioning-resnet
```

Create a virtual environment:

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

## Model Files

The trained model is stored in:

```text
models/final_model.pth
```

Vocabulary:

```text
config/vocabulary.json
```

Inference configuration:

```text
config/inference_config.json
```

The Flickr8k dataset itself is intentionally not included in the repository.

---

## Reproducibility

Important artifacts were saved separately during development, including:

- trained model checkpoints
- vocabulary
- inference configuration
- training history
- evaluation metrics
- generated captions
- qualitative failure cases

The project can therefore be inspected and run without retraining the baseline model.

---

## Limitations

The current architecture compresses the entire image into a single 512-dimensional feature vector before passing the representation to the LSTM decoder.

This creates a bottleneck for fine-grained spatial understanding.

The baseline therefore has difficulty with:

- object identity
- object counting
- object attributes
- object interactions
- visually similar objects
- detailed actions
- images substantially different from the training distribution

These limitations are intentionally documented rather than hidden.

---

## Future Work

Potential improvements include:

### 1. Spatial Attention

Instead of collapsing the CNN representation immediately into a single vector:

```text
Current:

ResNet
  ↓
Global Average Pooling
  ↓
512-D Vector
  ↓
LSTM
```

an attention-based extension could preserve spatial information:

```text
ResNet Feature Map
        ↓
Spatial Attention
        ↓
Weighted Visual Features
        ↓
LSTM Decoder
```

This would allow the decoder to focus on different regions of the image while generating different words.

### 2. Beam Search

The current inference implementation uses greedy decoding.

A future version can compare greedy decoding with beam search.

### 3. Larger Datasets

The system can be trained on larger datasets such as Flickr30k to investigate whether additional visual-language diversity improves generalization.

### 4. Improved Evaluation

Future experiments can include additional captioning metrics such as CIDEr and more systematic human evaluation.

### 5. Error-Driven Experiments

The qualitative failure cases provide specific directions for further experimentation:

```text
Wrong object
      ↓
Investigate visual representation

Wrong attribute
      ↓
Investigate spatial information

Wrong interaction
      ↓
Investigate multi-object relationships
```

---

## Research Direction

The current implementation can serve as a reproducible baseline for future experiments.

A natural next experiment is:

```text
Baseline

ResNet Encoder
      ↓
Global Feature Vector
      ↓
LSTM
      ↓
Caption
```

versus:

```text
Attention Extension

ResNet Feature Map
      ↓
Spatial Attention
      ↓
LSTM
      ↓
Caption
```

The experimental question becomes:

> Does preserving spatial visual information through attention improve image captioning compared with the global-feature baseline?

This provides a clear baseline → hypothesis → implementation → evaluation workflow.

---

## What This Project Demonstrates

This project demonstrates practical experience with:

- Convolutional neural networks
- Residual learning
- LSTM sequence modeling
- Word embeddings
- Teacher forcing
- Cross-entropy optimization
- Sequence padding
- Vocabulary construction
- Greedy decoding
- Model checkpointing
- Quantitative evaluation
- Qualitative error analysis
- PyTorch
- Streamlit
- Reproducible inference

---

## Responsible Reporting

The project reports the actual measured performance of the implemented baseline.

No claim of state-of-the-art performance is made.

The model's limitations and failure cases are explicitly documented as part of the project analysis.

---

## Author

**Munna Kumar Shah**

Computer Science and Engineering

GitHub:

https://github.com/doitmuna

---

## License

This project is intended for educational and research purposes.