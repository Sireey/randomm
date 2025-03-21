from config import Config
from model import VisionGPT2Model
from preprocessing import TeluguDataset, build_vocab, collate_fn, load_telugu_captions, train_tfms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from google.colab import drive
import sys
import os
import random
import pandas as pd
import numpy as np
import gc
from types import SimpleNamespace
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_linear_schedule_with_warmup


# Add project files to path
sys.path.append("/content/drive/MyDrive/Project_files")

# Import your modules


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
        self.current_epoch = 0

        # Initialize model
        self.model = VisionGPT2Model.from_pretrained(
            model_config).to(self.device)

        # Initially freeze pretrained layers
        self.model.pretrained_layers_trainable(trainable=False)

        print(
            f'Trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        self.train_dl, self.val_dl = dls

        # Initialize optimizer and scheduler
        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.train_config.lr,
            weight_decay=0.01
        )

        # Create a scheduler - linear warmup followed by cosine decay
        self.sched = get_linear_schedule_with_warmup(
            self.optim,
            # 2 epochs of warmup
            num_warmup_steps=len(self.train_dl) * 2 if self.train_dl else 100,
            num_training_steps=len(
                self.train_dl) * self.train_config.epochs if self.train_dl else 1000
        )

        # Initialize metrics dataframe with expanded metrics
        self.metrics = pd.DataFrame(columns=[
            'train_loss', 'train_perplexity', 'val_loss', 'val_perplexity',
            'lr', 'unfrozen_layers'
        ])

        # Define transformations for caption generation
        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

        # Create directory for model checkpoints
        os.makedirs(self.train_config.model_path, exist_ok=True)

        # Create attention maps directory if specified
        if hasattr(self.train_config, 'attention_maps_path'):
            os.makedirs(self.train_config.attention_maps_path, exist_ok=True)

        # Keep track of current lr and unfrozen layer info
        self.current_lr = self.train_config.lr
        self.unfrozen_layers = "None (Only projection layers)"

    def save_model(self, epoch=None):
        """Save model checkpoint"""
        if epoch is not None:
            save_path = os.path.join(
                self.train_config.model_path, f'captioner_epoch_{epoch}.pt')
        else:
            save_path = os.path.join(
                self.train_config.model_path, 'captioner_best.pt')

        # Save model state dict separately to reduce memory usage during saving
        model_state = self.model.state_dict()

        state_dict = {
            'model': model_state,
            'optimizer': self.optim.state_dict(),
            'scheduler': self.sched.state_dict(),
            'metrics': self.metrics,
            'epoch': self.current_epoch if epoch is None else epoch,  # Changed line
            'vocab_size': self.model_config.vocab_size
        }

        print(f"Saving checkpoint to {save_path}...")
        torch.save(state_dict, save_path)
        print(f"Model saved successfully")

    def load_checkpoint(self, path=None):
        """Load model checkpoint"""
        if path is None:
            path = os.path.join(
                self.train_config.model_path, 'captioner_best.pt')

        try:
            print(f"Loading checkpoint from {path}")
            # Load checkpoint to CPU first to avoid OOM on GPU
            checkpoint = torch.load(
                path, map_location='cpu', weights_only=False)

            # Load model state dict
            self.model.load_state_dict(checkpoint['model'])

            # Move model to proper device after loading
            self.model = self.model.to(self.device)

            # Load optimizer and scheduler states
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.sched.load_state_dict(checkpoint['scheduler'])

            # Load metrics
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']

            loaded_epoch = checkpoint.get('epoch', -1)
            print(f"Successfully loaded checkpoint (epoch {loaded_epoch})")
            return True, loaded_epoch
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False, -1

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
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)

            self.optim.step()
            self.sched.step()
            self.optim.zero_grad(set_to_none=True)

            running_loss += loss.item()

            prog.set_description(
                f'Epoch {epoch} | Train Loss: {loss.item():.3f} | LR: {self.optim.param_groups[0]["lr"]:.1e}')

            # Free up memory
            del image, input_ids, labels, loss
            self.clean()

        train_loss = running_loss / len(self.train_dl)
        # Cap perplexity to avoid overflow
        train_pxp = np.exp(min(train_loss, 20))

        # Get current learning rate
        current_lr = self.optim.param_groups[0]['lr']

        self.metrics.loc[epoch, ['train_loss', 'train_perplexity', 'lr', 'unfrozen_layers']] = (
            train_loss, train_pxp, current_lr, self.unfrozen_layers)

        self.current_epoch = epoch

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

            # Free up memory
            del image, input_ids, labels, loss
            self.clean()

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(min(val_loss, 20))  # Cap perplexity to avoid overflow

        self.metrics.loc[epoch, ['val_loss', 'val_perplexity']] = (
            val_loss, val_pxp)

        return val_loss, val_pxp

    def generate_sample_caption(self, image_path):
        """Generate caption for a sample image"""
        self.model.eval()

        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        transformed_image = self.gen_tfms(image=image_np)['image']
        image_tensor = transformed_image.unsqueeze(0).to(self.device)

        # Start with BOS token
        input_ids = torch.tensor(
            [[self.model_config.bos_token_id]]).to(self.device)

        # Generate text
        with torch.no_grad():
            caption_ids = self.model.generate(
                image_tensor,
                input_ids,
                max_tokens=20,
                temperature=1.2,
                repetition_penalty=1.5,
                deterministic=False
            )

            print("Generation process:")
            for i in range(caption_ids.shape[1]):
                token_id = caption_ids[0, i].item()  # Get token ID
                token_word = idx_to_word.get(
                    token_id, "<unk>")  # Get corresponding word
                print(
                    f"Step {i}: Selected token ID {token_id} → '{token_word}'")

        # Convert IDs back to tokens
        dataset = self.train_dl.dataset if self.train_dl else None
        if dataset is None and self.val_dl:
            dataset = self.val_dl.dataset

        if dataset is None:
            # If we don't have dataset information, we can't convert IDs to tokens
            return "[No vocabulary available to decode caption]"

        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        idx_to_word = {idx: word for word, idx in dataset.vocab.items()}

        tokens = [idx_to_word.get(idx, "<unk>") for idx in caption_ids.squeeze().tolist()
                  if idx not in [0, 1, 2, 3]]  # Skip special tokens

        caption = " ".join(tokens)
        return caption

    @torch.no_grad()
    def visualize_attention(self, image_path, epoch):
        """
        Generate caption and visualize attention weights
        Returns the generated caption
        """
        self.model.eval()

        # Check if attention_maps_path is set
        if not hasattr(self.train_config, 'attention_maps_path'):
            print("Attention maps path not set. Skipping attention visualization.")
            return self.generate_sample_caption(image_path)

        # Load and process image
        try:
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)
            transformed_image = self.gen_tfms(image=original_image)['image']
            image_tensor = transformed_image.unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return f"[Error loading image: {str(e)}]"

        # Start with BOS token
        input_ids = torch.tensor(
            [[self.model_config.bos_token_id]]).to(self.device)

        try:
            # Check if generate_with_attention method exists in the model
            if hasattr(self.model, 'generate_with_attention'):
                # Generate text with attention weights
                caption_ids, attention_weights = self.model.generate_with_attention(
                    image_tensor,
                    input_ids,
                    max_tokens=20,
                    temperature=1.0,
                    deterministic=True
                )
            else:
                # Fall back to regular generation if attention visualization is not available
                print(
                    "Model doesn't support attention visualization. Using regular generation.")
                caption_ids = self.model.generate(
                    image_tensor,
                    input_ids,
                    max_tokens=20,
                    temperature=1.0,
                    deterministic=True
                )
                return self.generate_sample_caption(image_path)
        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"[Error generating caption: {str(e)}]"

        # Convert IDs back to tokens
        dataset = self.train_dl.dataset if self.train_dl else None
        if dataset is None and self.val_dl:
            dataset = self.val_dl.dataset

        if dataset is None:
            # If we don't have dataset information, we can't convert IDs to tokens
            return "[No vocabulary available to decode caption]"

        if isinstance(dataset, Subset):
            dataset = dataset.dataset
        idx_to_word = {idx: word for word, idx in dataset.vocab.items()}

        tokens = [idx_to_word.get(idx, "<unk>") for idx in caption_ids.squeeze().tolist()
                  if idx not in [0, 1, 2, 3]]  # Skip special tokens

        caption = " ".join(tokens)

        # Now create attention visualization if we have attention weights
        try:
            if 'attention_weights' in locals() and attention_weights is not None:
                img_name = os.path.basename(image_path).split('.')[0]
                epoch_str = str(epoch)
                save_dir = os.path.join(
                    self.train_config.attention_maps_path, f"epoch_{epoch_str}")
                os.makedirs(save_dir, exist_ok=True)

                # Save attention maps
                for i, attn in enumerate(attention_weights):
                    # Reshape attention for visualization
                    # This depends on your model's architecture
                    # Example: Reshape and normalize attention
                    attn_map = attn.cpu().numpy()

                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.imshow(original_image)
                    ax.imshow(attn_map, cmap='hot', alpha=0.5)
                    ax.axis('off')

                    # Add token information if available
                    if i < len(tokens):
                        ax.set_title(f"Attention for token: {tokens[i]}")

                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        save_dir, f"{img_name}_token_{i}.png"))
                    plt.close()
        except Exception as e:
            print(f"Error visualizing attention: {e}")

        return caption

    def clean(self):
        """Clean memory"""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def plot_training_metrics(self):
        """Plot training and validation metrics"""
        if len(self.metrics) < 2:
            print("Not enough data to plot metrics")
            return

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))

        # Plot loss
        axs[0, 0].plot(self.metrics['train_loss'], label='Train Loss')
        axs[0, 0].plot(self.metrics['val_loss'], label='Val Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot perplexity
        axs[0, 1].plot(self.metrics['train_perplexity'],
                       label='Train Perplexity')
        axs[0, 1].plot(self.metrics['val_perplexity'], label='Val Perplexity')
        axs[0, 1].set_title('Perplexity')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Perplexity')
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot learning rate
        axs[1, 0].plot(self.metrics['lr'])
        axs[1, 0].set_title('Learning Rate')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Learning Rate')
        axs[1, 0].set_yscale('log')
        axs[1, 0].grid(True)

        # Plot unfrozen layers info as text
        axs[1, 1].axis('off')
        unfrozen_text = "Unfrozen Layers by Epoch:\n\n"
        for i, layer_info in enumerate(self.metrics['unfrozen_layers']):
            unfrozen_text += f"Epoch {i}: {layer_info}\n"
        axs[1, 1].text(0.1, 0.5, unfrozen_text, fontsize=10, va='center')

        plt.tight_layout()

        # Save figure
        metrics_path = os.path.join(
            self.train_config.model_path, 'training_metrics.png')
        plt.savefig(metrics_path)
        plt.close()

        print(f"Training metrics plot saved to {metrics_path}")

    def generate_val_captions(self, epoch, num_samples=5):
        """Generate captions for a few validation samples"""
        if self.val_dl is None or len(self.val_dl.dataset) == 0:
            print("No validation samples available")
            return

        val_size = len(self.val_dl.dataset)
        sample_indices = random.sample(
            range(val_size), min(num_samples, val_size))

        print(f"\n=== Validation Captions at Epoch {epoch} ===")

        captions = []
        for idx in sample_indices:
            try:
                sample = self.val_dl.dataset[idx]

                # Get image path from dataset
                if isinstance(self.val_dl.dataset, Subset):
                    img_path = self.val_dl.dataset.dataset.df.iloc[
                        self.val_dl.dataset.indices[idx]]['image']
                    true_caption = self.val_dl.dataset.dataset.df.iloc[
                        self.val_dl.dataset.indices[idx]]['caption']
                else:
                    img_path = self.val_dl.dataset.df.iloc[idx]['image']
                    true_caption = self.val_dl.dataset.df.iloc[idx]['caption']

                # Generate caption with attention if possible
                gen_caption = self.visualize_attention(img_path, epoch)

                print(f"Image: {os.path.basename(img_path)}")
                print(f"True caption: {true_caption}")
                print(f"Generated: {gen_caption}")
                print("-" * 50)

                captions.append({
                    'image': os.path.basename(img_path),
                    'true_caption': true_caption,
                    'generated_caption': gen_caption
                })
            except Exception as e:
                print(f"Error processing validation sample {idx}: {e}")

        # Save captions to file
        try:
            captions_path = os.path.join(
                self.train_config.model_path, f'val_captions_epoch_{epoch}.json')

            import json
            with open(captions_path, 'w', encoding='utf-8') as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)

            print(f"Validation captions saved to {captions_path}")
        except Exception as e:
            print(f"Error saving validation captions: {e}")

    def fit(self, start_epoch=0):
        """Training loop with gradual unfreezing and validation caption generation"""
        best_loss = float('inf')
        if not self.metrics.empty and 'val_loss' in self.metrics:
            # Find the previous best loss
            min_loss = self.metrics['val_loss'].min()
            if not pd.isna(min_loss):
                best_loss = min_loss
                print(f"Previous best validation loss: {best_loss:.4f}")

        best_epoch = -1
        epochs_range = range(start_epoch, self.train_config.epochs)
        prog = tqdm(epochs_range, desc="Epochs")

        # Define unfreezing schedule (more gradual)
        unfreezing_schedule = {
            # Starting point: No unfreezing
            0: ("None (Only projection layers)", 1.0),
            # Unfreeze last GPT2 block
            2: ("Last GPT2 block", 0.8),
            # Unfreeze last 2 GPT2 blocks
            4: ("Last 2 GPT2 blocks", 0.5),
            # Unfreeze all GPT2 blocks
            6: ("All GPT2 blocks", 0.3),
            8: ("All layers", 0.1)                     # Unfreeze everything
        }

        for epoch in prog:
            # Check if this epoch needs layer unfreezing according to schedule
            if epoch in unfreezing_schedule:
                layer_desc, lr_factor = unfreezing_schedule[epoch]
                self.unfrozen_layers = layer_desc

                # Adjust model layers according to unfreezing stage
                if layer_desc == "Last GPT2 block":
                    self.model.unfreeze_last_gpt_block()
                    print(f'Unfreezing last GPT2 block...')
                elif layer_desc == "Last 2 GPT2 blocks":
                    self.model.unfreeze_last_n_gpt_blocks(2)
                    print(f'Unfreezing last 2 GPT2 blocks...')
                elif layer_desc == "All GPT2 blocks":
                    self.model.unfreeze_gpt_layers()
                    print(f'Unfreezing all GPT2 layers...')
                elif layer_desc == "All layers":
                    self.model.pretrained_layers_trainable(trainable=True)
                    print(f'Unfreezing all layers...')

                # Update learning rate based on schedule factor
                new_lr = self.train_config.lr * lr_factor
                print(f'Adjusting learning rate to {new_lr:.1e}')

                # Re-initialize optimizer with updated learning rate
                self.optim = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=new_lr,
                    weight_decay=0.01
                )

                # Re-initialize scheduler
                remaining_epochs = self.train_config.epochs - epoch
                self.sched = get_linear_schedule_with_warmup(
                    self.optim,
                    num_warmup_steps=len(self.train_dl),  # 1 epoch warmup
                    num_training_steps=len(self.train_dl) * remaining_epochs
                )

                # Update current learning rate tracking
                self.current_lr = new_lr

            try:
                # Training phase
                train_loss = self.train_one_epoch(epoch)
                self.clean()

                # Validation phase
                val_loss, val_ppl = self.valid_one_epoch(epoch)
                self.clean()

                # Generate validation captions and visualize attention maps every 2 epochs
                if epoch % 2 == 0 or epoch == self.train_config.epochs - 1:
                    self.generate_val_captions(epoch)

                # Save checkpoint every epoch
                self.save_model(epoch=epoch)

                # Plot and save metrics
                self.plot_training_metrics()

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
            except Exception as e:
                print(f"Error during epoch {epoch}: {e}")
                # Try to save current model state in case of error
                try:
                    self.save_model(epoch=f"{epoch}_error")
                except:
                    pass

        # Final attention maps visualization on best model
        print("\nGenerating final attention maps with best model...")
        try:
            self.load_checkpoint()  # Load best model
            self.generate_val_captions(epoch="final", num_samples=10)
        except Exception as e:
            print(f"Error generating final visualizations: {e}")

        return {
            'best_loss': best_loss,
            'best_perplexity': np.exp(min(best_loss, 20)),  # Cap perplexity
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image_path, max_tokens=50, temperature=1.0, deterministic=False):
        """Generate caption for an image"""
        self.model.eval()

        try:
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
            dataset = self.train_dl.dataset if self.train_dl else None
            if dataset is None and self.val_dl:
                dataset = self.val_dl.dataset

            if dataset is None:
                # If we don't have dataset information, we can't convert IDs to tokens
                return "[No vocabulary available to decode caption]"

            # Handle Subset case
            if isinstance(dataset, Subset):
                dataset = dataset.dataset
            idx_to_word = {idx: word for word, idx in dataset.vocab.items()}

            tokens = [idx_to_word.get(idx, "<unk>") for idx in caption_ids.squeeze().tolist()
                      if idx not in [0, 1, 2, 3]]  # Skip special tokens

            caption = " ".join(tokens)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return f"[Error: {str(e)}]"


def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Mount Google Drive if not already mounted
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(
            f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(
            f"CUDA Memory Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("WARNING: CUDA not available. Using CPU instead.")
        device = torch.device('cpu')

    # Configuration
    model_config = Config()

    # Your specific paths
    data_path = "/content/drive/MyDrive/randomm/Data/fl8telugu.txt"
    images_folder = "/content/drive/MyDrive/randomm/Data/Images"

    # Define training configuration
    train_config = SimpleNamespace(
        epochs=10,  # Increased from 5 to 10
        lr=2e-5,
        device=device,
        model_path='/content/drive/MyDrive/randomm/telugu_captioner_checkpoints',
        attention_maps_path='/content/drive/MyDrive/randomm/attention_maps',
        batch_size=4,
        num_workers=2
    )

    # Check if files exist
    if not os.path.exists(data_path):
        print(f"ERROR: Caption file not found at {data_path}")
        return

    if not os.path.exists(images_folder):
        print(f"ERROR: Images folder not found at {images_folder}")
        return

    # Prepare dataset with reduced samples for faster experimentation
    train_ds, val_ds, vocab = prepare_reduced_data(
        caption_path=data_path,
        image_dir=images_folder,
        max_samples=500,  # Limit to 500 samples
        val_split=0.1
    )

    # Update vocab size in model config
    model_config.vocab_size = len(vocab)
    print(f"Vocabulary size: {model_config.vocab_size}")

    # Create data loaders
    train_dl = DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize trainer
    trainer = Trainer(model_config, train_config, (train_dl, val_dl))

    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_path = os.path.join(
        train_config.model_path, 'captioner_best.pt')
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}")
        success, loaded_epoch = trainer.load_checkpoint(checkpoint_path)
        if success and loaded_epoch >= 0:
            start_epoch = loaded_epoch + 1
            print(f"Resuming training from epoch {start_epoch}")

    # Train the model
    try:
        results = trainer.fit(start_epoch=start_epoch)
        print("\nTraining complete!")
        print(f"Best validation loss: {results['best_loss']:.4f}")
        print(f"Best perplexity: {results['best_perplexity']:.4f}")
        print(f"Best epoch: {results['best_epoch']}")

        # Final evaluation
        print("\nGenerating sample captions with the best model...")
        sample_images = [
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)[:5]
            if img.endswith(('.jpg', '.jpeg', '.png'))
        ]

        for img_path in sample_images:
            try:
                caption = trainer.generate_caption(
                    img_path,
                    max_tokens=30,
                    temperature=0.7
                )
                print(f"Image: {os.path.basename(img_path)}")
                print(f"Caption: {caption}")
                print("-" * 50)
            except Exception as e:
                print(f"Error generating caption for {img_path}: {e}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        trainer.save_model(epoch="interrupted")
    except Exception as e:
        print(f"Error during training: {e}")
        # Save model in case of error
        trainer.save_model(epoch="error")

    print("Done!")


if __name__ == "__main__":
    main()
