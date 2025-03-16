from PIL import Image
from indicnlp.tokenize import indic_tokenize
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from collections import Counter
import os
import random
from torch.utils.data import Dataset


def load_telugu_captions(filepath):
    """Load Telugu captions from the dataset file"""
    captions_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            img_name = parts[0].split("#")[0]
            caption = parts[1]

            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)
    return captions_dict


def build_vocab(df, min_freq=2):
    """Build vocabulary from Telugu captions"""
    word_freq = Counter()

    # Tokenize captions and count word occurrences
    for caption in df["caption"]:
        tokens = indic_tokenize.trivial_tokenize(caption, lang='te')
        word_freq.update(tokens)

    # Keep only words that appear at least `min_freq` times
    vocab_words = [word for word, freq in word_freq.items()
                   if freq >= min_freq]

    # Sort vocabulary for consistency
    vocab_words.sort()

    # Create vocab mapping
    vocab = {}

    # Add special tokens first
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    vocab["<s>"] = 2
    vocab["</s>"] = 3

    # Add regular tokens
    for idx, word in enumerate(vocab_words, start=4):
        vocab[word] = idx

    print(f"Vocabulary size: {len(vocab)} words")
    return vocab


class TeluguDataset(Dataset):
    """Dataset for Telugu image captioning"""

    def __init__(self, df, tfms, vocab):
        self.df = df
        self.tfms = tfms
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image_path = sample["image"]
        caption = sample["caption"]

        # Handle file not found errors gracefully
        try:
            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            image = self.tfms(image=image)["image"]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros(3, 224, 224)

        # Process caption using IndicNLP tokenizer
        tokens = indic_tokenize.trivial_tokenize(caption, lang='te')

        # Convert tokens to indices
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["<unk>"])

        # Add special tokens
        token_ids = [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]

        # Create labels (shifted by 1)
        labels = token_ids.copy()
        labels[:-1] = token_ids[1:]
        labels[-1] = -100  # Ignore last token for loss calculation

        return image, token_ids, labels


def collate_fn(batch):
    """Collate function for DataLoader"""
    # Extract images, input_ids, and labels from the batch
    images = []
    input_ids = []
    labels = []

    for item in batch:
        # Skip problematic samples
        if item is None or len(item) != 3:
            continue

        image, ids, lbls = item
        images.append(image)
        input_ids.append(ids)
        labels.append(lbls)

    if len(images) == 0:
        # Return empty batch if all samples were problematic
        return torch.zeros(0, 3, 224, 224), torch.zeros(0, 1).long(), torch.zeros(0, 1).long()

    # Stack images into a single tensor (batch dimension)
    images = torch.stack(images, dim=0)

    # Pad input_ids (captions) and labels to the longest sequence in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids],
                                                batch_first=True,
                                                padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in labels],
                                             batch_first=True,
                                             padding_value=-100)  # Use -100 for padding in labels

    return images, input_ids, labels


# Define Transformations
train_tfms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ColorJitter(p=0.3),
    A.Affine(scale=(0.8, 1.2), translate_percent=(
        0.1, 0.1), rotate=(-15, 15), p=0.5),
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])


def visualize_dataset_sample(dataset, idx=None):
    """Visualize a sample from the dataset"""
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    image, token_ids, labels = dataset[idx]

    # Convert indices back to tokens
    idx_to_word = {idx: word for word, idx in dataset.vocab.items()}
    tokens = [idx_to_word.get(idx, "<unk>") for idx in token_ids]

    # Print sample info
    print(f"Sample {idx}:")
    print(f"Image shape: {image.shape}")
    print(f"Token IDs: {token_ids}")
    print(f"Caption: {''.join(tokens[1:-1])}")  # Skip special tokens

    # Display image
    img = image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
    img = (img * 0.5) + 0.5  # De-normalize

    # Plot image with matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Caption: {''.join(tokens[1:-1])}")
    plt.axis('off')
    plt.show()


# sample
# 1. Define paths to your data
# Update this with your actual file path
data_path = "D:\\ict\\Data\\fl8telugu.txt"
# Update this with your actual folder path
images_folder = "D:\\ict\\Data\\Images"

# 2. Load just a few captions for testing
print("Loading sample captions for testing...")
captions_dict = {}
with open(data_path, "r", encoding="utf-8") as f:
    # Just read the first 5 lines for testing
    for i, line in enumerate(f):
        if i >= 5:  # Adjust this number to test with more or fewer samples
            break

        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        img_name = parts[0].split("#")[0]
        caption = parts[1]

        if img_name not in captions_dict:
            captions_dict[img_name] = []
        captions_dict[img_name].append(caption)

print(
    f"Sample captions loaded: {sum(len(caps) for caps in captions_dict.values())} captions for {len(captions_dict)} images")

# 3. Create a small DataFrame for testing
data = []
for img_name, captions in captions_dict.items():
    img_path = os.path.join(images_folder, img_name)
    if os.path.exists(img_path):
        for caption in captions:
            data.append({"image": img_path, "caption": caption})
    else:
        print(f"Warning: Image {img_path} not found")

df = pd.DataFrame(data)
print(f"Test dataset size: {len(df)} image-caption pairs")

if len(df) == 0:
    print("No valid image-caption pairs found. Check your file paths.")
else:
    # 4. Build vocabulary from this small sample
    # Use min_freq=1 for small test samples
    vocab = build_vocab(df, min_freq=1)

    # 5. Create test dataset
    dataset = TeluguDataset(df, train_tfms, vocab)

    # 6. Print sample details without visualization
    idx = 0
    image, token_ids, labels = dataset[idx]

    # Convert indices back to tokens
    idx_to_word = {idx: word for word, idx in dataset.vocab.items()}
    tokens = [idx_to_word.get(idx, "<unk>") for idx in token_ids]

    print(f"\nSample {idx}:")
    print(f"Image shape: {image.shape}")
    print(f"Token IDs: {token_ids}")
    print(f"Caption: {''.join(tokens[1:-1])}")  # Skip special tokens

    # Print first few processed images in batch
    batch = [dataset[i] for i in range(min(3, len(dataset)))]
    images, input_ids, label_ids = collate_fn(batch)
    print(f"\nBatch shape: {images.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Label IDs shape: {label_ids.shape}")
