import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from preprocessing import TeluguDataset, build_vocab, collate_fn, load_telugu_captions, train_tfms
from model import VisionGPT2Model
from config import Config
from pathlib import Path
import pandas as pd
import numpy as np
import gc
from types import SimpleNamespace
from PIL import Image
import os
import random


def prepare_reduced_data(caption_path, image_dir, max_samples=500, val_split=0.1):
    """Prepare reduced training and validation datasets"""
    print(f"Loading Telugu captions from {caption_path}...")
    # Load all captions first
    telugu_captions = load_telugu_captions(caption_path)

    # Get a list of all image names
    all_images = list(telugu_captions.keys())

    # If we have more images than max_samples, randomly sample max_samples images
    if len(all_images) > max_samples:
        print(
            f"Randomly sampling {max_samples} images from {len(all_images)} total images")
        sampled_images = random.sample(all_images, max_samples)
        # Create a reduced dictionary with only the sampled images
        reduced_captions = {
            img: telugu_captions[img] for img in sampled_images}
    else:
        print(
            f"Using all {len(all_images)} images (less than requested {max_samples})")
        reduced_captions = telugu_captions

    # Count total caption-image pairs
    total_pairs = sum(len(captions) for captions in reduced_captions.values())
    print(
        f"Selected {len(reduced_captions)} images with {total_pairs} total captions")

    # Create DataFrame from the reduced dataset
    data = []
    for img, captions in reduced_captions.items():
        img_path = os.path.join(image_dir, img)
        if os.path.exists(img_path):
            for caption in captions:
                data.append({"image": img_path, "caption": caption})
        else:
            print(f"Warning: Image {img_path} not found")

    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} valid image-caption pairs")

    # Build vocabulary
    vocab = build_vocab(df, min_freq=1)  # Lower min_freq for small datasets

    # Create dataset
    dataset = TeluguDataset(df, train_tfms, vocab)

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    return train_ds, val_ds, vocab


class Trainer:
    def __init__(self, model_config, train_config, dls):
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        # Initialize model
        self.model = VisionGPT2Model.from_pretrained(
            model_config).to(self.device)

        # Initially freeze pretrained layers
        self.model.pretrained_layers_trainable(trainable=False)

        print(
            f'Trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        self.train_dl, self.val_dl = dls

        # Optimizer with weight decay
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.lr / 25.0,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=len(self.train_dl),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=10000.0
        )

        # Track metrics
        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity',
                      'val_loss', 'val_perplexity']] = None

        # Transforms for inference
        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

    def save_model(self, epoch=None):
        """Save model checkpoint"""
        self.train_config.model_path.mkdir(exist_ok=True)

        if epoch is not None:
            save_path = self.train_config.model_path / \
                f'captioner_epoch_{epoch}.pt'
        else:
            save_path = self.train_config.model_path / 'captioner_best.pt'

        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.sched.state_dict(),
            'metrics': self.metrics
        }

        torch.save(state_dict, save_path)
        print(f"Model saved to {save_path}")

    def load_best_model(self):
        """Load the best checkpoint"""
        try:
            checkpoint = torch.load(
                self.train_config.model_path / 'captioner_best.pt',
                map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            print("Loaded best model checkpoint")
            return True
        except:
            print("Failed to load checkpoint")
            return False

    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        prog = tqdm(self.train_dl, total=len(self.train_dl))
        running_loss = 0.

        for image, input_ids, labels in prog:
            image = image.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            loss = self.model(image, input_ids, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optim.step()
            self.sched.step()
            self.optim.zero_grad(set_to_none=True)

            running_loss += loss.item()

            prog.set_description(
                f'Epoch {epoch} | Train Loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        # Cap perplexity to avoid overflow
        train_pxp = np.exp(min(train_loss, 20))

        self.metrics.loc[epoch, ['train_loss', 'train_perplexity']] = (
            train_loss, train_pxp)
        return train_loss

    @torch.no_grad()
    def valid_one_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        prog = tqdm(self.val_dl, total=len(self.val_dl))
        running_loss = 0.

        for image, input_ids, labels in prog:
            image = image.to(self.device)
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            loss = self.model(image, input_ids, labels)
            running_loss += loss.item()

            prog.set_description(
                f'Epoch {epoch} | Valid Loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(min(val_loss, 20))  # Cap perplexity to avoid overflow

        self.metrics.loc[epoch, ['val_loss', 'val_perplexity']] = (
            val_loss, val_pxp)

        return val_loss, val_pxp

    def clean(self):
        """Clean memory"""
        gc.collect()

    def fit(self):
        """Training loop"""
        best_loss = float('inf')
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs), desc="Epochs")

        for epoch in prog:
            # Progressive unfreezing based on epochs
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('Unfreezing GPT2 layers...')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)
                print('Unfreezing all layers...')

            # Training phase
            train_loss = self.train_one_epoch(epoch)
            self.clean()

            # Validation phase
            val_loss, val_ppl = self.valid_one_epoch(epoch)
            self.clean()

            # Save checkpoint every few epochs
            if (epoch + 1) % 2 == 0:  # Save more frequently on small dataset
                self.save_model(epoch=epoch)

            # Print metrics
            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_ppl:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                print(
                    f'New best model at epoch {epoch} with validation loss: {val_loss:.4f}')
                self.save_model()

        return {
            'best_loss': best_loss,
            'best_perplexity': np.exp(min(best_loss, 20)),  # Cap perplexity
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image_path, max_tokens=50, temperature=1.0, deterministic=False):
        """Generate caption for an image"""
        self.model.eval()

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)

        # Start with BOS token
        sequence = torch.tensor(
            [[self.model_config.bos_token_id]]).to(self.device)

        # Generate caption
        caption_ids = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )

        # Convert IDs back to tokens
        idx_to_word = {idx: word for word,
                       idx in self.train_dl.dataset.dataset.vocab.items()}
        tokens = [idx_to_word.get(idx, "<unk>") for idx in caption_ids.tolist()
                  if idx not in [0, 1, 2, 3]]  # Skip special tokens

        caption = " ".join(tokens)
        return caption


def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Set up device - use CPU
    device = torch.device('cpu')
    print(f"Using device: {device}")

    # Configuration
    model_config = Config()

    train_config = SimpleNamespace(
        epochs=5,               # Keep original 5 epochs
        freeze_epochs_gpt=2,
        freeze_epochs_all=3,
        lr=2e-5,
        device=device,
        model_path=Path('telugu_captioner_test'),
        batch_size=4,
        num_workers=0
    )

    # Prepare reduced data - just 100 images for testing
    caption_path = "D:/ict/Data/fl8telugu.txt"
    image_dir = "D:/ict/Data/Images/"

    train_ds, val_ds, vocab = prepare_reduced_data(
        caption_path, image_dir, max_samples=100, val_split=0.1)

    model_config.vocab_size = len(vocab)
    print(f"Vocabulary size: {model_config.vocab_size}")

    model_config.depth = 12  # Full GPT2 depth for pretrained compatibility

    train_dl = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    trainer = Trainer(model_config, train_config, (train_dl, val_dl))

    # Check for existing checkpoint
    checkpoint_path = Path('telugu_captioner_test') / 'captioner_best.pt'
    if checkpoint_path.exists():
        print(f"Found checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optim.load_state_dict(checkpoint['optimizer'])
        trainer.sched.load_state_dict(checkpoint['scheduler'])
        trainer.metrics = checkpoint['metrics']

        print("Running final epoch (5)...")
        epoch = 4  # 0-indexed

        train_loss = trainer.train_one_epoch(epoch)
        trainer.clean()

        val_loss, val_ppl = trainer.valid_one_epoch(epoch)
        trainer.clean()

        print(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_ppl:.4f}")

        trainer.save_model(epoch=epoch)

        best_loss = trainer.metrics['val_loss'].min()
        if val_loss < best_loss:
            print(
                f'New best model at epoch {epoch} with validation loss: {val_loss:.4f}')
            trainer.save_model()  # Save as best model

        print("Training completed successfully!")

    else:
        print("No checkpoint found, running full training...")
        results = trainer.fit()
        print(f"Training completed. Best results: {results}")

    # Generate captions for a few validation images
    try:
        print("\nGenerating sample captions...")
        for i in range(min(3, len(val_ds))):
            sample = val_ds[i]
            if isinstance(val_ds, Subset):
                sample_idx = val_ds.indices[i]
                img_path = val_ds.dataset.df.iloc[sample_idx]['image']
            else:
                img_path = val_ds.df.iloc[i]['image']

            caption = trainer.generate_caption(img_path)
            print(f"Image: {os.path.basename(img_path)}")
            print(f"Generated caption: {caption}")
    except Exception as e:
        print(f"Error generating captions: {e}")


if __name__ == "__main__":
    main()
