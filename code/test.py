import os
import glob
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np

# Import modules
from model import ZeroDCEPlusPlus
from dataloader import device

# ==============================
# Model Loading (Priority: default > final)
# ==============================
model = ZeroDCEPlusPlus(num_iter=8).to(device)

default_path = 'zero_dce_pp_trained_default.pth'
final_path = 'zero_dce_pp_final.pth'

if os.path.exists(default_path):
    checkpoint_path = default_path
    print(f"Loading default pre-trained model: {default_path}")
elif os.path.exists(final_path):
    checkpoint_path = final_path
    print(f"Loading your trained model: {final_path}")
else:
    raise FileNotFoundError(
        "Neither 'zero_dce_pp_trained_default.pth' nor 'zero_dce_pp_final.pth' found.\n"
        "Please place one of these files in the current directory."
    )

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Model loaded and set to evaluation mode.\n")

# ==============================
# Dataset Paths
# ==============================
try:
    test_low_dir = glob.glob('**/eval15/low', recursive=True)[0]
    test_high_dir = glob.glob('**/eval15/high', recursive=True)[0]
except IndexError:
    raise FileNotFoundError("Test folders 'eval15/low' or 'eval15/high' not found. Make sure LOL dataset is extracted.")

low_paths = sorted([os.path.join(test_low_dir, f) for f in os.listdir(test_low_dir) if f.lower().endswith('.png')])
high_paths = sorted([os.path.join(test_high_dir, f) for f in os.listdir(test_high_dir) if f.lower().endswith('.png')])

print(f"Found {len(low_paths)} test image pairs.\n")

full_res_transform = transforms.ToTensor()


# ==============================
# Main Functions
# ==============================
def visualize_single(index: int):
    if index < 0 or index >= len(low_paths):
        raise ValueError(f"Index must be between 0 and {len(low_paths)-1}")

    low_path = low_paths[index]
    high_path = high_paths[index]
    filename = os.path.basename(low_path)

    print(f"Visualizing test image {index}: {filename}")

    # Load images
    low_img = Image.open(low_path).convert('RGB')
    high_img = Image.open(high_path).convert('RGB')

    low_tensor = full_res_transform(low_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        enhanced_tensor, _ = model(low_tensor)
        enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)

    # To numpy
    low_np = low_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    high_np = full_res_transform(high_img).permute(1, 2, 0).numpy()
    enhanced_np = enhanced_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # Plot 1: Input vs Ground Truth
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(low_np)
    plt.title('Input: Low-Light (400×600)', fontsize=15)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(high_np)
    plt.title('Ground Truth: Normal-Light', fontsize=15)
    plt.axis('off')

    plt.suptitle(f'Zero-DCE++ Result — Test Index {index}: {filename}', fontsize=18)
    plt.tight_layout()
    plt.show()

    # Plot 2: Enhanced only (large)
    plt.figure(figsize=(10, 12))
    plt.imshow(enhanced_np)
    plt.title(f'Enhanced by Zero-DCE++\nIndex {index}: {filename}\nFull Resolution (400×600)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Single image visualization complete for index {index}.\n")


def visualize_all():
    print("Running inference on all test images and creating comparison grid...")

    low_tensors = []
    enhanced_tensors = []
    high_tensors = []
    filenames = []

    with torch.no_grad():
        for idx, (low_path, high_path) in enumerate(zip(low_paths, high_paths)):
            filename = os.path.basename(low_path)
            filenames.append(filename)

            low_img = Image.open(low_path).convert('RGB')
            high_img = Image.open(high_path).convert('RGB')

            low_tensor = full_res_transform(low_img).unsqueeze(0).to(device)
            enhanced_tensor, _ = model(low_tensor)
            enhanced_tensor = torch.clamp(enhanced_tensor, 0.0, 1.0)

            low_tensors.append(low_tensor.squeeze(0).cpu())
            enhanced_tensors.append(enhanced_tensor.squeeze(0).cpu())
            high_tensors.append(full_res_transform(high_img))

    print("Generating grid...")

    padding = 10
    row_grids = []
    for i in range(len(low_paths)):
        row = [low_tensors[i], high_tensors[i], enhanced_tensors[i]]
        grid = make_grid(row, nrow=3, padding=padding, pad_value=1.0)
        row_grids.append(grid)

    grid_np_list = [g.permute(1, 2, 0).numpy() for g in row_grids]
    final_image = np.vstack(grid_np_list)

    plt.figure(figsize=(24, 3.8 * len(low_paths)))
    plt.imshow(final_image)
    plt.axis('off')
    plt.title('All Test Images: Low-Light → Ground Truth → Zero-DCE++ Enhanced',
              fontsize=22, pad=40)

    # Add labels
    h_per_row = final_image.shape[0] // len(low_paths)
    w_per_col = final_image.shape[1] // 3
    for i in range(len(low_paths)):
        y = i * h_per_row + 30
        plt.text(20, y, f'{i+1:2d}: {filenames[i]}', fontsize=13, color='black', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=6))
        plt.text(w_per_col + 20, y, 'Ground Truth', fontsize=13, color='black', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=6))
        plt.text(2 * w_per_col + 20, y, 'Enhanced', fontsize=13, color='black', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=6))

    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.06)
    plt.show()

    print(f"All {len(low_paths)} test images displayed successfully!")


# ==============================
# Argument Parser
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Zero-DCE++ results on LOL test set")
    parser.add_argument('--index', type=int, default=6,
                        help='Index of single test image to display (0-14). Default: 6')
    parser.add_argument('--all', action='store_true',
                        help='Display all 15 test images in a comparison grid')

    args = parser.parse_args()

    if args.all:
        visualize_all()
    else:
        visualize_single(args.index)