from preprocessing import SentencePieceTokenizer
from model import VisionGPT2Model, ModelConfig
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.append('/content/drive/MyDrive/Project_files/')


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for the model

    Args:
        image_path (str): Path to the image file

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Define image transformations (match your training preprocessing)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size to match model input
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Open image
    image = Image.open(image_path).convert('RGB')

    # Apply transformations
    return transform(image).unsqueeze(0)  # Add batch dimension


def load_model_for_inference(checkpoint_path, tokenizer_path):
    """
    Load a trained model for inference

    Args:
        checkpoint_path (str): Path to the model checkpoint
        tokenizer_path (str): Path to the SentencePiece tokenizer model

    Returns:
        tuple: (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    # Recreate model configuration
    model_config = ModelConfig()
    model_config.vocab_size = tokenizer.vocab_size
    model_config.eos_token_id = tokenizer.eos_id
    model_config.depth = checkpoint['config'].get('model_depth', 12)

    # Initialize model
    model = VisionGPT2Model.from_pretrained(model_config)

    # Load model state dict
    model.load_state_dict(checkpoint['model'])

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return model, tokenizer


def inference(model, tokenizer, image_tensor, max_tokens=50, temperature=0.7):
    """
    Generate caption for a single image

    Args:
        model (VisionGPT2Model): Trained model
        tokenizer (SentencePieceTokenizer): Tokenizer
        image_tensor (torch.Tensor): Preprocessed image tensor
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature

    Returns:
        str: Generated caption
    """
    device = next(model.parameters()).device

    # Move image to device
    image_tensor = image_tensor.to(device)

    # Use BOS token as input for generation
    input_ids = torch.tensor([[tokenizer.bos_id]]).to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            image_tensor,
            input_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=False
        )[0].cpu().numpy().tolist()

    # Decode caption
    try:
        # Remove tokens after EOS
        if tokenizer.eos_id in generated_ids:
            generated_ids = generated_ids[:generated_ids.index(
                tokenizer.eos_id)]
        generated_caption = tokenizer.decode(generated_ids)
    except Exception as e:
        generated_caption = "[Error decoding tokens]"

    return generated_caption


def run_inference(checkpoint_path, tokenizer_path, image_path):
    """
    Run inference on a single image

    Args:
        checkpoint_path (str): Path to model checkpoint
        tokenizer_path (str): Path to SentencePiece tokenizer model
        image_path (str): Path to image for captioning
    """
    # Load model and tokenizer
    model, tokenizer = load_model_for_inference(
        checkpoint_path, tokenizer_path)

    # Preprocess image
    image_tensor = load_and_preprocess_image(image_path)

    # Generate caption
    caption = inference(model, tokenizer, image_tensor)

    # Visualize
    plt.figure(figsize=(10, 5))

    # Display image
    plt.subplot(1, 2, 1)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    # Display caption
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, caption,
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=12,
             wrap=True)
    plt.title("Generated Caption")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Generated Caption:", caption)


# Example usage
if __name__ == "__main__":
    # Configure these paths
    CHECKPOINT_PATH = "/content/drive/MyDrive/randomm/best_model_1.pt"
    # Path to your SentencePiece model
    TOKENIZER_PATH = "/content/drive/MyDrive/randomm/tokenizer.model"
    IMAGE_PATH = "/content/drive/MyDrive/input.jpg"  # Replace with your image path

    # Run inference
    run_inference(CHECKPOINT_PATH, TOKENIZER_PATH, IMAGE_PATH)
