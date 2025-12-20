import os
import glob
import zipfile
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==============================
# Device Setup
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")

# ==============================
# Dataset Path Detection & Auto-Extraction
# ==============================
print("\nSearching for LOL dataset...")

train_low_candidates = glob.glob('**/our485/low', recursive=True)
test_low_candidates  = glob.glob('**/eval15/low', recursive=True)

if train_low_candidates and test_low_candidates:
    train_low_dir = train_low_candidates[0]
    test_low_dir  = test_low_candidates[0]
    print("Dataset folders already found.")
else:
    possible_zip_names = ['lol-dataset.zip', 'archive.zip']
    zip_path = None
    for name in possible_zip_names:
        if os.path.exists(name):
            zip_path = name
            break

    if zip_path is None:
        raise FileNotFoundError(
            "\nLOL dataset not found and no zip file detected.\n\n"
            "Please:\n"
            "1. Download from: https://www.kaggle.com/datasets/soumikrakshit/lol-dataset\n"
            "2. Place the zip file (lol-dataset.zip or archive.zip) in this folder\n"
            "3. Re-run the script – it will extract automatically."
        )

    print(f"\nFound {zip_path} – extracting now...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('.')
    print("Extraction complete.\n")

    train_low_candidates = glob.glob('**/our485/low', recursive=True)
    test_low_candidates  = glob.glob('**/eval15/low', recursive=True)

    if not train_low_candidates or not test_low_candidates:
        raise RuntimeError(
            "Extraction finished, but dataset folders not found. "
            "Make sure you downloaded the correct LOL dataset from Kaggle."
        )

    train_low_dir = train_low_candidates[0]
    test_low_dir  = test_low_candidates[0]

train_high_dir = train_low_dir.replace('low', 'high')
test_high_dir  = test_low_dir.replace('low', 'high')

if not os.path.exists(train_high_dir) or not os.path.exists(test_high_dir):
    raise FileNotFoundError("High-resolution folders missing. Dataset is incomplete.")

print(f"Train low : {train_low_dir} ({len(os.listdir(train_low_dir))} images)")
print(f"Train high: {train_high_dir} ({len(os.listdir(train_high_dir))} images)")
print(f"Test low  : {test_low_dir} ({len(os.listdir(test_low_dir))} images)")
print(f"Test high : {test_high_dir} ({len(os.listdir(test_high_dir))} images)")


# ==============================
# Dataset Class & DataLoaders (unchanged except num_workers)
# ==============================
class LOLDataset(Dataset):

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomCrop(256, pad_if_needed=True),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

train_dataset = LOLDataset(mode='train', transform=train_transform)
test_dataset  = LOLDataset(mode='test',  transform=test_transform)

# Reduced num_workers for better compatibility (especially Windows)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"\nDataLoaders ready:")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches : {len(test_loader)}")

# ==============================
# Visualization (unchanged)
# ==============================
# ... (keep your visualization code exactly as is)