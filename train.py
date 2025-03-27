import os
import time
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.append('/content/drive/MyDrive/Codes/')
from preprocessing import (
    prepare_telugu_captioning_datasets,
    collate_fn,
    SentencePieceTokenizer
)

# Import from model
from model import VisionGPT2Model, ModelConfig

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configure these paths ---
DATA_PATH = "/content/drive/MyDrive/Data/fl8telugu.txt"
IMAGES_FOLDER = "/content/drive/MyDrive/Data/Images"
TOKENIZER_PATH = "/content/drive/MyDrive/Data/tokenizer.model"
OUTPUT_DIR = "/content/drive/MyDrive/training_output"
# ---------------------------

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training configuration
class TrainingConfig:
    def __init__(self):
        self.max_samples = None  # Limited sample size for initial training
        self.batch_size = 8
        self.num_workers = 2
        self.num_epochs = 10
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.grad_clip_val = 1.0
        self.teacher_forcing_ratio = 0.8  # Probability of using teacher forcing
        self.teacher_forcing_decay = 0.95  # Decay teacher forcing over epochs
        self.use_mixed_precision = True  # Use mixed precision training
        self.save_every = 1  # Save checkpoint every N epochs
        self.eval_every = 50  # Evaluate every N steps
        self.warmup_steps = 100
        self.model_depth = 12  # Number of transformer layers
        self.log_interval = 20  # Log training stats every N steps

        # Set devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1

def log_metrics(metrics, step, epoch=None, is_train=True, tb_writer=None):
    """Log training/validation metrics to console and tensorboard"""
    prefix = "Train" if is_train else "Val"
    epoch_str = f" Epoch {epoch}" if epoch is not None else ""

    # Print to console
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"{prefix}{epoch_str} Step {step} | {metric_str}")

    # Log to tensorboard if writer is provided
    if tb_writer is not None:
        for k, v in metrics.items():
            tb_writer.add_scalar(f"{prefix}/{k}", v, step)

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, loss, config, tokenizer, filename):
    """Save training checkpoint"""
    if config.multi_gpu:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'teacher_forcing_ratio': config.teacher_forcing_ratio,
        'vocab_size': tokenizer.vocab_size,
        'config': {k: v for k, v in vars(config).items() if not k.startswith('__') and not callable(v)}
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def visualize_sample(model, tokenizer, val_dataset, device, idx=None, save_path=None):
    """Visualize a prediction on a sample from the validation dataset"""
    if idx is None:
        idx = random.randint(0, len(val_dataset) - 1)

    # Get sample data
    image, token_ids, _ = val_dataset[idx]
    original_caption = val_dataset.df.iloc[idx]['caption']

    # Generate caption
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        # Use only BOS token as input for generation
        input_ids = torch.tensor([[tokenizer.bos_id]]).to(device)
        generated_ids = model.generate(
            image,
            input_ids,
            max_tokens=50,
            temperature=0.7,
            deterministic=False
        )[0].cpu().numpy().tolist()

    # Decode generated caption
    try:
        # Find where EOS token appears
        if tokenizer.eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(tokenizer.eos_id)]
        generated_caption = tokenizer.decode(generated_ids)
    except:
        generated_caption = "[Error decoding tokens]"

    # Print results
    print(f"Original: {original_caption}")
    print(f"Generated: {generated_caption}")

    # Convert and display image
    img = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img * std + mean
    img = np.clip(img, 0, 1)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Generated: {generated_caption}")
    plt.figtext(0.5, 0.01, f"Original: {original_caption}", wrap=True,
                horizontalalignment='center', fontsize=10)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    model.train()
    return generated_caption

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, config, scaler=None):
    """
    Load a training checkpoint and return the starting epoch, global step, and other states

    Args:
        checkpoint_path (str): Path to the checkpoint file
        model (nn.Module): Model to load state dict into
        optimizer (Optimizer): Optimizer to load state dict
        scheduler (LRScheduler): Learning rate scheduler to load state dict
        config (TrainingConfig): Training configuration object
        scaler (GradScaler, optional): Gradient scaler for mixed precision training

    Returns:
        dict: Dictionary containing training state information
    """
    checkpoint = torch.load(checkpoint_path)

    # Load model state
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load scheduler state if exists
    if scheduler and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Load scaler state for mixed precision training
    if scaler and checkpoint['scaler']:
        scaler.load_state_dict(checkpoint['scaler'])

    # Return key training states
    return {
        'start_epoch': checkpoint['epoch'],
        'global_step': checkpoint['step'],
        'best_val_loss': checkpoint.get('loss', float('inf')),
        'teacher_forcing_ratio': checkpoint.get('teacher_forcing_ratio', config.teacher_forcing_ratio)
    }

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory
    """
    # List all .pt files in the directory
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]

    if not checkpoints:
        return None

    # Sort checkpoints by epoch number
    latest_checkpoint = max(checkpoints, key=lambda x:
        int(x.split('checkpoint_epoch')[1].split('.pt')[0])
    )

    return os.path.join(checkpoint_dir, latest_checkpoint)

def train(resume_from_checkpoint=None):
    # Initialize configuration
    config = TrainingConfig()
    if resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint('/content/drive/MyDrive/training_output')
    set_seed(42)

    # Load tokenizer
    print(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = SentencePieceTokenizer(TOKENIZER_PATH)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, _ = prepare_telugu_captioning_datasets(
        data_path=DATA_PATH,
        images_folder=IMAGES_FOLDER,
        tokenizer_path=TOKENIZER_PATH,
        val_split=0.1,
        max_samples=config.max_samples
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Calculate total steps for learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs

    # Initialize model
    print("Initializing model...")
    model_config = ModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    model_config.eos_token_id = tokenizer.eos_id
    model_config.depth = config.model_depth

    model = VisionGPT2Model.from_pretrained(model_config, device=config.device)

    # Unfreeze necessary layers for fine-tuning
    model.pretrained_layers_trainable(False)  # First freeze all
    model.unfreeze_word_embeddings()  # Unfreeze embeddings for our tokenizer
    model.unfreeze_last_n_gpt_blocks(3)  # Unfreeze last 3 GPT blocks

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")

    # Move model to device and setup multi-GPU if available
    if config.multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Initialize optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )

    # Initialize grad scaler for mixed precision training
    scaler = GradScaler() if config.use_mixed_precision else None

    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    start_epoch = 0

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
      print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
      resume_states = load_checkpoint(
        resume_from_checkpoint,
        model,
        optimizer,
        scheduler,
        config,
        scaler
    )
    start_epoch = resume_states['start_epoch']
    global_step = resume_states['global_step']
    best_val_loss = resume_states['best_val_loss']

    # Restore teacher forcing ratio from checkpoint
    config.teacher_forcing_ratio = resume_states['teacher_forcing_ratio']

    print(f"Resuming from Epoch {start_epoch}, Global Step {global_step}")
    print(f"Previous Best Validation Loss: {best_val_loss:.4f}")
    print(f"Restored Teacher Forcing Ratio: {config.teacher_forcing_ratio:.4f}")

    try:
        for epoch in range(config.num_epochs):
            # Update teacher forcing ratio (decay over epochs)
            if epoch > 0:
                config.teacher_forcing_ratio *= config.teacher_forcing_decay

            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print(f"Teacher forcing ratio: {config.teacher_forcing_ratio:.4f}")

            # Training phase
            model.train()
            epoch_loss = 0
            start_time = time.time()

            train_iterator = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for step, batch in enumerate(train_iterator):
                # Unpack batch
                images, token_ids, labels, attention_mask = batch

                # Skip empty batches
                if images.size(0) == 0:
                    continue

                # Move batch to device
                images = images.to(config.device)
                token_ids = token_ids.to(config.device)
                labels = labels.to(config.device)
                attention_mask = attention_mask.to(config.device)

                # Forward pass with teacher forcing
                optimizer.zero_grad()

                # Use mixed precision if enabled
                if config.use_mixed_precision:
                    with autocast():
                        loss = model(images, token_ids, attention_mask, labels)

                    # Scale gradients and optimize
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = model(images, token_ids, attention_mask, labels)
                    loss.backward()
                    clip_grad_norm_(model.parameters(), config.grad_clip_val)
                    optimizer.step()

                # Update learning rate
                scheduler.step()

                # Update metrics
                global_step += 1
                epoch_loss += loss.item()

                # Log progress
                if step % config.log_interval == 0 or step == len(train_loader) - 1:
                    avg_loss = epoch_loss / (step + 1)
                    examples_per_sec = (step + 1) * config.batch_size / (time.time() - start_time)
                    lr = optimizer.param_groups[0]['lr']

                    metrics = {
                        'loss': loss.item(),
                        'avg_loss': avg_loss,
                        'lr': lr,
                        'examples/sec': examples_per_sec
                    }
                    log_metrics(metrics, global_step, epoch+1)

                    # Update progress bar
                    train_iterator.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.6f}"
                    })

                # Evaluate periodically
                if global_step % config.eval_every == 0:
                    print("\nGenerating sample caption...")
                    sample_output_path = os.path.join(OUTPUT_DIR, f"sample_epoch{epoch+1}_step{global_step}.png")
                    visualize_sample(
                        model.module if config.multi_gpu else model,
                        tokenizer,
                        val_dataset,
                        config.device,
                        save_path=sample_output_path
                    )

            # Validation phase
            model.eval()
            val_loss = 0
            val_steps = 0

            print("\nValidating...")
            with torch.no_grad():
                for val_batch in tqdm(val_loader, desc="Validation"):
                    # Unpack batch
                    val_images, val_token_ids, val_labels, val_attention_mask = val_batch

                    # Skip empty batches
                    if val_images.size(0) == 0:
                        continue

                    # Move batch to device
                    val_images = val_images.to(config.device)
                    val_token_ids = val_token_ids.to(config.device)
                    val_labels = val_labels.to(config.device)
                    val_attention_mask = val_attention_mask.to(config.device)

                    # Forward pass
                    batch_loss = model(val_images, val_token_ids, val_attention_mask, val_labels)
                    val_loss += batch_loss.item()
                    val_steps += 1

            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % config.save_every == 0:
                checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
                save_checkpoint(
                    model.module if config.multi_gpu else model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    global_step,
                    avg_val_loss,
                    config,
                    tokenizer,
                    checkpoint_path
                )

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
                save_checkpoint(
                    model.module if config.multi_gpu else model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch + 1,
                    global_step,
                    best_val_loss,
                    config,
                    tokenizer,
                    best_model_path
                )
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

            # Generate sample captions
            print("\nGenerating sample captions...")
            for _ in range(3):
                visualize_sample(
                    model.module if config.multi_gpu else model,
                    tokenizer,
                    val_dataset,
                    config.device
                )

    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        interrupt_path = os.path.join(OUTPUT_DIR, "interrupt_checkpoint.pt")
        save_checkpoint(
            model.module if config.multi_gpu else model,
            optimizer,
            scheduler,
            scaler,
            epoch + 1,
            global_step,
            epoch_loss / (step + 1),
            config,
            tokenizer,
            interrupt_path
        )

    print("Training completed!")

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, "final_model.pt")
    save_checkpoint(
        model.module if config.multi_gpu else model,
        optimizer,
        scheduler,
        scaler,
        config.num_epochs,
        global_step,
        avg_val_loss,
        config,
        tokenizer,
        final_model_path
    )

    return model, tokenizer

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")

    # Run training
    model, tokenizer = train()