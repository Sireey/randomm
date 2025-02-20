from PIL import Image
from indicnlp.tokenize import indic_tokenize
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from collections import Counter

def load_telugu_captions(filepath):
    captions_dict = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, caption = parts[0].split("#")[0], parts[1]
            if img_name not in captions_dict:
                captions_dict[img_name] = []
            captions_dict[img_name].append(caption)
    return captions_dict

# Convert text to Unicode IDs (Simple Tokenizer Workaround)
def text_to_ids(text):
    tokens = indic_tokenize.trivial_tokenize(text, lang='te')
    token_ids = [ord(char) for token in tokens for char in token]  # Unicode ID mapping
    return token_ids

class TeluguDataset:
    def __init__(self, df, tfms, vocab):
        self.df = df
        self.tfms = tfms
        self.vocab = vocab  # Dictionary mapping words to indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image_path = sample["image"]
        caption = sample["caption"]

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = self.tfms(image=image)["image"]

        # Process caption using IndicNLP tokenizer
        tokens = indic_tokenize.trivial_tokenize(caption, lang='te')  
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

        # Add special tokens
        token_ids = [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]

        labels = token_ids.copy()
        labels[:-1] = token_ids[1:]

        return image, token_ids, labels

def build_vocab(df, min_freq=2):
    word_freq = Counter()

    # Tokenize captions and count word occurrences
    for caption in df["caption"]:
        tokens = indic_tokenize.trivial_tokenize(caption, lang='te')
        word_freq.update(tokens)

    # Keep only words that appear at least `min_freq` times
    vocab_words = [word for word, freq in word_freq.items() if freq >= min_freq]

    # Create vocab mapping
    vocab = {word: idx for idx, word in enumerate(vocab_words, start=4)}

    # Add special tokens
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    vocab["<s>"] = 2
    vocab["</s>"] = 3

    return vocab

def collate_fn(batch):
    # Extract images, input_ids, and labels from the batch
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]

    # Stack images into a single tensor (batch dimension)
    image = torch.stack(image, dim=0)

    # Pad input_ids (captions) and labels to the longest sequence in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in labels], batch_first=True, padding_value=0)

    # Create a mask for real tokens (not padding)
    mask = (input_ids != 0).long()

    # Set padding tokens in labels to -100 (to ignore during loss calculation)
    labels[mask == 0] = -100

    return image, input_ids, labels

caption_path = "D:/ict/Data/fl8telugu.txt"
telugu_captions = load_telugu_captions(caption_path)

# Convert to DataFrame
df = pd.DataFrame(
    [{"image": f"D:/ict/Data/Images/{img}", "caption": caption} for img, captions in telugu_captions.items() for caption in captions]
)

# Define Transformations
train_tfms = A.Compose([
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
    ToTensorV2()
])

vocab = build_vocab(df, min_freq=2)
print("Vocabulary Size:", len(vocab))

dataset = TeluguDataset(df, train_tfms, vocab)

# Test Preprocessing
sample_image, sample_input_ids, sample_labels = dataset[0]

print("Sample Image Shape:", sample_image.shape)
print("Sample Token IDs:", sample_input_ids)
print("Sample Labels:", sample_labels)

idx_to_word = {idx: word for word, idx in vocab.items()}
tokens = [idx_to_word[idx] for idx in sample_input_ids]
print("Decoded Caption:", " ".join(tokens))

import matplotlib.pyplot as plt
import torchvision.transforms as T

img = sample_image.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
img = (img * 0.5) + 0.5  # De-normalize (assuming mean=0.5, std=0.5)
plt.imshow(img)
plt.show()