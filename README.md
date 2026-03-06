# Image Caption Generator

A deep learning pipeline that automatically generates natural language descriptions for images, built with TensorFlow/Keras. The system uses a CNN-LSTM encoder-decoder architecture to learn the mapping between visual features and textual descriptions.

## Architecture Overview

The model follows a **merge architecture** composed of two main branches:

```
Image --> ResNet50 --> Dense(256) --\
                                    --> add --> Dense(256) --> Dense(vocab_size, softmax)
Caption --> Embedding(256) --> LSTM(256) --/
```

### Image Encoder: ResNet50 (pretrained)

- **Model**: ResNet50 with ImageNet weights, `include_top=False`, global average pooling
- **Output**: a 2048-dimensional feature vector per image
- **Why ResNet50**: It provides a strong balance between depth and computational cost. The residual connections solve the vanishing gradient problem that affects very deep networks, allowing it to learn rich visual representations. Compared to VGG16 (commonly used in captioning tutorials), ResNet50 is deeper yet faster due to its bottleneck blocks, and produces more discriminative features. Using `include_top=False` with `pooling='avg'` removes the classification head and gives a compact 2048-d embedding, ideal for downstream fusion with text.
- **Why pretrained (transfer learning)**: Training a CNN from scratch on Flickr8k (~8,000 images) would massively overfit. Leveraging ImageNet features means the encoder already understands edges, textures, objects, and scenes, which transfers well to captioning.

### Caption Decoder: Embedding + LSTM

- **Embedding layer**: 256-dimensional, with `mask_zero=True` to handle variable-length sequences padded with zeros
- **LSTM**: single layer, 256 units
- **Why LSTM over vanilla RNN**: LSTMs use gating mechanisms (input, forget, output gates) that allow the network to selectively remember or forget information over long sequences. Vanilla RNNs suffer from vanishing gradients and struggle to capture dependencies beyond a few timesteps — a critical limitation when generating multi-word captions. LSTMs are the standard choice for sequence generation tasks for this reason.
- **Why 256 units**: A good trade-off for a dataset of this size. Larger hidden states (512, 1024) risk overfitting on Flickr8k; smaller ones may not capture enough language structure.

### Decoder (Merge Layer)

- The image features and the LSTM output are projected to the same dimensionality (256) and combined with **element-wise addition**
- A final Dense layer with **softmax** activation predicts the next word from the vocabulary
- **Why merge (add) instead of inject**: In the "inject" approach, the image is fed as the first hidden state of the LSTM, coupling vision and language processing. The merge approach keeps the two modalities independent until the final decoding step, which has been shown to produce more stable training and comparable or better results on small-to-medium datasets.

## Training Strategy

| Choice | Detail | Rationale |
|---|---|---|
| **Loss** | `categorical_crossentropy` | Standard for multi-class word prediction — each timestep is a classification over the vocabulary |
| **Optimizer** | Adam | Adaptive learning rates per parameter; converges faster than SGD and requires less hyperparameter tuning |
| **EarlyStopping** | `patience=2`, `restore_best_weights=True` | Prevents overfitting by halting training when the loss stops improving, and automatically reverts to the best model |
| **Dropout (0.5)** | Applied on both image features and embedding output | Strong regularization to prevent overfitting on the relatively small Flickr8k dataset |
| **Batch size** | 32 | Standard default that fits comfortably in GPU memory while providing stable gradient estimates |
| **Max epochs** | 50 | Upper bound — in practice EarlyStopping triggers well before this |

### Data Generator

Training data is generated on the fly via a Python generator to avoid loading the entire expanded dataset into memory. For each image-caption pair, the caption is split into incremental input-output subsequences (teacher forcing), where at each step the model sees the ground truth prefix and learns to predict the next word.

## Training Results

### Early Stopping with patience=2

<img width="719" height="446" alt="image" src="https://github.com/user-attachments/assets/7c6ca25b-1ea2-4190-86d6-521e115470ad" />


With `patience=2`, early stopping triggers after just **6 epochs**. The training loss decreases steadily from ~0.059 to ~0.040, while the validation loss drops sharply from ~0.130 to ~0.069 in the first two epochs but then starts rising again around epoch 4-5 (reaching ~0.075). This divergence between training and validation loss is a clear sign of **overfitting**: the model starts memorizing training examples rather than learning generalizable patterns. Early stopping correctly halts training before the gap widens further, and `restore_best_weights=True` reverts the model to the epoch with the lowest validation loss.

### Early Stopping with patience=5

<img width="723" height="449" alt="image" src="https://github.com/user-attachments/assets/0fc791b9-75ef-40e2-b2a9-e4b65cdc907f" />

With a higher `patience=5`, the model trains for **11 epochs**, giving it more time to converge. The validation loss reaches a lower minimum (~0.053 around epoch 4-5) compared to the patience=2 run, because the model is allowed to push through short plateaus. However, after epoch 8 the validation loss begins creeping upward again (~0.065), confirming that overfitting eventually sets in regardless. The extra epochs provide a modest improvement in validation performance, but with diminishing returns — the gap between training loss (~0.040) and validation loss (~0.055) remains, reflecting the limited size of the Flickr8k dataset. A patience of 2 is ultimately a reasonable and conservative choice that avoids unnecessary computation while still capturing most of the learning.

## Text Preprocessing

- Captions are lowercased and stripped of special characters to reduce vocabulary size
- Special tokens `startseq` and `endseq` are prepended/appended to each caption to signal the beginning and end of generation
- A `Tokenizer` with an `<unk>` OOV token is fitted on all training captions

## Inference

Caption generation uses **greedy decoding**: starting from `startseq`, the model predicts the most probable next word at each step, appends it to the sequence, and repeats until `endseq` is predicted or the maximum length is reached.

### Sample Output

<img width="1387" height="768" alt="image" src="https://github.com/user-attachments/assets/d0051096-ed95-4bb3-b77b-716fecf5010d" />

The image shows people at a dock pulling ropes near a ship, but the model generates: *"a man in a white shirt is driving a car"*. While the model correctly identifies the presence of a person and a white shirt, it fails to capture the actual scene context (a port, ropes, multiple people). This is a typical limitation of models trained on small datasets like Flickr8k: the model learns frequent co-occurring patterns from training (e.g., "man" + "white shirt" often appears with "car" in captions) and falls back on these statistical shortcuts rather than truly understanding the scene. More diverse training data, attention mechanisms, or beam search decoding could help improve accuracy.

## Dataset

The project uses the **Flickr8k** dataset, which contains ~8,000 images each annotated with 5 human-written captions. It is a standard benchmark for image captioning, small enough to train on a single GPU yet diverse enough to learn meaningful vision-language associations.

## Project Structure

```
image_caption_generator/
├── image_caption_generator.py   # Full pipeline: feature extraction, training, inference
├── captions.txt                 # Image-caption pairs (CSV format)
├── images/                      # Raw image files (Flickr8k)
├── features/                    # Pre-extracted ResNet50 feature vectors (.npy)
├── assets/                      # Training plots and sample outputs
└── README.md
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- tqdm

## How to Run

1. **Place the Flickr8k dataset** — extract `images/` and `captions.txt` into the project root.
2. **Run the script**:
   ```bash
   python image_caption_generator.py
   ```
   On the first run, ResNet50 features are extracted and cached in `features/`. Subsequent runs skip this step.
3. After training completes, the script generates a sample caption and prints it to the console.
