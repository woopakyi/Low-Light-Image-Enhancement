import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import cv2

# Import modules
from model import ZeroDCEPlusPlus
from dataloader import device


# ==============================
# Load Enhancement Model
# ==============================
def load_enhancement_model():
    model = ZeroDCEPlusPlus(num_iter=8).to(device)

    default_path = 'zero_dce_pp_trained_default.pth'
    final_path = 'zero_dce_pp_final.pth'

    if os.path.exists(default_path):
        checkpoint = default_path
        print(f"✓ Loading author's pre-trained model: {default_path}")
    elif os.path.exists(final_path):
        checkpoint = final_path
        print(f"✓ Loading your trained model: {final_path}")
    else:
        raise FileNotFoundError(
            "No enhancement model found!\n"
            "Please provide:\n"
            "  • zero_dce_pp_trained_default.pth (recommended – author's pre-trained)\n"
            "  • zero_dce_pp_final.pth (from train.py)"
        )

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


# ==============================
# Generate Enhanced Image
# ==============================
def enhance_image(model, low_path):
    low_pil = Image.open(low_path).convert('RGB')
    low_tensor = transforms.ToTensor()(low_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor, _ = model(low_tensor)
        enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)

    enhanced_np = enhanced_tensor[0].cpu().permute(1, 2, 0).numpy()
    low_np = low_tensor[0].cpu().permute(1, 2, 0).numpy()
    return low_np, enhanced_np


# ==============================
# Apply Spotlight Effect
# ==============================
def apply_spotlight(enhanced_np, subject_mask):
    spotlight = enhanced_np.copy()

    # Darken background
    spotlight[subject_mask == 0] *= 0.25

    # Brighten & boost contrast on subject
    subject_pixels = spotlight[subject_mask == 1]
    subject_pixels = np.clip(subject_pixels * 1.15 + 0.1, 0, 1)
    spotlight[subject_mask == 1] = subject_pixels

    return spotlight


# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply spotlight effect to low-light image using mask")
    parser.add_argument('--index', type=int, default=6,
                        help="Test image index (0–14). Default: 6")
    parser.add_argument('--mask-path', type=str, default=None,
                        help="Optional: Path to pre-generated mask (e.g., subject_mask.npy). If not provided, generates one.")
    args = parser.parse_args()

    # Load enhancement model
    enhancement_model = load_enhancement_model()

    # Find test image
    try:
        test_low_dir = glob.glob('**/eval15/low', recursive=True)[0]
    except IndexError:
        raise FileNotFoundError("Test folder 'eval15/low' not found. Extract LOL dataset first.")

    low_paths = sorted([
        os.path.join(test_low_dir, f)
        for f in os.listdir(test_low_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if args.index < 0 or args.index >= len(low_paths):
        raise ValueError(f"Index must be between 0 and {len(low_paths)-1}")

    low_path = low_paths[args.index]
    filename = os.path.basename(low_path)
    print(f"\nProcessing: {filename} (index {args.index})\n")

    # Enhance image
    low_np, enhanced_np = enhance_image(enhancement_model, low_path)

    # Load or generate mask
    if args.mask_path and os.path.exists(args.mask_path):
        print(f"Loading pre-generated mask from {args.mask_path}")
        subject_mask = np.load(args.mask_path)
    else:
        print("No pre-generated mask provided → generating one...")
        # Import and use mask_generator from maskloader.py if available
        try:
            from maskloader import load_mobile_sam, generate_subject_mask
            mask_generator = load_mobile_sam()
            subject_mask = generate_subject_mask(enhanced_np, mask_generator)
        except ImportError:
            raise ImportError(
                "Mask generation requires maskloader.py in the same folder.\n"
                "Or provide --mask-path to a .npy mask file."
            )

    # Apply spotlight
    print("Applying spotlight effect...")
    spotlight_result = apply_spotlight(enhanced_np, subject_mask)

    # Visualize
    plt.figure(figsize=(12, 10))
    plt.imshow(spotlight_result)
    plt.title(f"Spotlight Effect: {filename}\n"
              "(Main subject brightly highlighted, background darkened)",
              fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("Spotlight effect complete! 🎭")