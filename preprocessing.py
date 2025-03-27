import os
import random
import pandas as pd
import numpy as np
import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- EDIT THESE PATHS FOR COLAB ---
DATA_PATH = "/content/drive/MyDrive/Data/fl8telugu.txt"  # Path to your captions file
IMAGES_FOLDER = "/content/drive/MyDrive/Data/Images"  # Path to your images folder
# Path to your tokenizer model
TOKENIZER_PATH = "/content/drive/MyDrive/Data/tokenizer.model"
# ---------------------------------


def load_telugu_captions(filepath):
    """Load Telugu captions from the dataset file"""
    captions_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            # Handling potential '#' in image names
            img_name = parts[0].split("#")[0]
            caption = parts[1]

            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)
    return captions_dict


class SentencePieceTokenizer:
    """Wrapper for SentencePiece tokenizer"""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()

        # Create special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()

        # Create index-to-token mapping for debugging and visualization
        self.idx_to_token = {i: self.sp.id_to_piece(
            i) for i in range(self.vocab_size)}

    def encode(self, text):
        """Encode text to token IDs"""
        # Return list of token IDs
        return self.sp.encode(text)

    def decode(self, ids):
        """Decode token IDs back to text"""
        # Filter out special tokens that should not be in the output
        return self.sp.decode(ids)

    def encode_with_special_tokens(self, text):
        """Encode with BOS and EOS special tokens added"""
        # Add BOS and EOS tokens
        return [self.bos_id] + self.sp.encode(text) + [self.eos_id]


class TeluguCaptioningDataset(Dataset):
    """Dataset for Telugu image captioning using SentencePiece tokenizer"""

    def __init__(self, df, tfms, tokenizer):
        self.df = df
        self.tfms = tfms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image_path = sample["image"]
        caption = sample["caption"]

        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
            image = self.tfms(image=image)["image"]
        except FileNotFoundError:
            print(
                f"Error loading image {image_path}: File not found. Skipping sample.")
            return None  # Return None to skip this sample in collate_fn

        token_ids = self.tokenizer.encode_with_special_tokens(caption)
        labels = token_ids[1:] + [self.tokenizer.pad_id]

        return image, torch.tensor(token_ids), torch.tensor(labels)


def collate_fn(batch):
    """Collate function for DataLoader with padding to handle variable sequence lengths"""
    # Extract images, input_ids, and labels from the batch
    images = []
    all_token_ids = []
    all_labels = []

    for item in batch:
        # Skip problematic samples
        if item is None or len(item) != 3:
            continue

        image, token_ids, labels = item
        images.append(image)
        all_token_ids.append(token_ids)
        all_labels.append(labels)

    if len(images) == 0:
        # Return empty batch if all samples were problematic
        return torch.zeros(0, 3, 224, 224), torch.zeros(0, 1).long(), torch.zeros(0, 1).long(), torch.zeros(0, 1).long()

    # Stack images into a single tensor (batch dimension)
    images = torch.stack(images, dim=0)

    # Determine maximum sequence length in this batch for both token_ids and labels
    max_len_ids = max(ids.size(0) for ids in all_token_ids)
    max_len_labels = max(lbl.size(0) for lbl in all_labels)

    # Create padded tensors for token_ids and labels
    padded_ids = torch.zeros(len(all_token_ids), max_len_ids, dtype=torch.long)
    # Use -100 for padding in labels (ignored by CrossEntropyLoss by default)
    padded_labels = torch.full(
        (len(all_labels), max_len_labels), -100, dtype=torch.long)

    # Create attention masks (1 for tokens, 0 for padding)
    attention_masks = torch.zeros(
        len(all_token_ids), max_len_ids, dtype=torch.long)

    # Fill padded tensors with actual values and create masks
    for i, (ids, labels) in enumerate(zip(all_token_ids, all_labels)):
        ids_len = ids.size(0)
        labels_len = labels.size(0)

        # Copy token IDs and set mask
        padded_ids[i, :ids_len] = ids
        attention_masks[i, :ids_len] = 1

        # Copy labels
        padded_labels[i, :labels_len] = labels

    return images, padded_ids, padded_labels, attention_masks


# Define transformations with Albumentations
train_tfms = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.1, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])


def prepare_telugu_captioning_datasets(data_path=DATA_PATH, images_folder=IMAGES_FOLDER,
                                       tokenizer_path=TOKENIZER_PATH,
                                       val_split=0.1, max_samples=None):
    """
    Prepare datasets for Telugu image captioning
    """
    print(f"Loading captions from {data_path}...")
    captions_dict = load_telugu_captions(data_path)

    print(f"Loading SentencePiece tokenizer from {tokenizer_path}...")
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    data = []
    for img_name, captions in captions_dict.items():
        img_path = os.path.join(images_folder, img_name)
        if os.path.exists(img_path):
            for caption in captions:
                data.append({"image": img_path, "caption": caption})
        else:
            print(f"Warning: Image {img_path} not found")

    df = pd.DataFrame(data)
    print(f"Total dataset size: {len(df)} image-caption pairs")

    if max_samples and max_samples < len(df):
        df = df.sample(max_samples, random_state=42)
        print(f"Limited dataset to {len(df)} samples for testing")

    shuffled_df = df.sample(frac=1, random_state=42)  # shuffled only once.
    val_size = int(len(shuffled_df) * val_split)
    train_df = shuffled_df.iloc[val_size:]
    val_df = shuffled_df.iloc[:val_size]

    print(
        f"Train set: {len(train_df)} samples, Validation set: {len(val_df)} samples")

    train_dataset = TeluguCaptioningDataset(train_df, train_tfms, tokenizer)
    val_dataset = TeluguCaptioningDataset(val_df, val_tfms, tokenizer)

    return train_dataset, val_dataset, tokenizer


def visualize_dataset_sample(dataset, idx=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Visualize a sample from the dataset with tokenized key-value pairs for Colab."""
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    image, token_ids, labels = dataset[idx]

    if image is None:
        return

    original_caption = dataset.df.iloc[idx]['caption']

    tokenized_pairs = {}
    for token_id in token_ids.tolist():
        token = dataset.tokenizer.sp.id_to_piece(token_id)
        tokenized_pairs[token_id] = token

    print(f"Sample {idx}:")
    print(f"Image shape: {image.shape}")
    print(f"Original caption: {original_caption}")
    print(f"Tokenized key-value pairs:")
    for token_id, token in tokenized_pairs.items():
        print(f"  {token_id}: {token}")

    img = image.permute(1, 2, 0).numpy()
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Caption: {original_caption}")
    plt.axis('off')
    plt.show()


# Example usage for Colab
if __name__ == "__main__":
    # Install required libraries if they're not already installed
    # !pip install albumentations sentencepiece

    # Prepare datasets
    train_dataset, val_dataset, tokenizer = prepare_telugu_captioning_datasets(
        data_path=DATA_PATH,
        images_folder=IMAGES_FOLDER,
        tokenizer_path=TOKENIZER_PATH,
        val_split=0.1,
        max_samples=1000  # For testing, set to None for full dataset
    )

    # Display sample
    try:
        visualize_dataset_sample(train_dataset)
    except Exception as e:
        print(f"Visualization error: {e}")

    # Test batch creation
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Inspect a batch
    images, token_ids, labels, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Token IDs: {token_ids.shape}")
    print(f"Labels: {labels.shape}")
    print(f"Attention masks: {masks.shape}")
