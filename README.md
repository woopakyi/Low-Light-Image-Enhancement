# Low-Light-Image-Enhancement
Deep CNN project for enhancing low-light images with Zero-DCE++ and MobileSAM. Restores visibility/color, enables creative spotlight &amp; bokeh effects, and uses semi-supervised training on the LOL dataset.


# dataloader.py =============================================
# Features:
- Extracts the dataset from the zip file
- Finds training (485 pairs) and test (15 pairs) folders
- Applies standard transforms
- Creates ready-to-use PyTorch DataLoaders
- Shows sample images

# Requirements:
pip install torch torchvision torchaudio matplotlib tqdm pillow


This project uses the LOL (Low-Light) dataset from Kaggle:
https://www.kaggle.com/datasets/soumikrakshit/lol-dataset

How to Download and Place the Dataset:
1. Go to the link above and download the file lol-dataset.zip
(You may need to sign in to Kaggle.)
2. Extract the zip file directly into the root folder of this project (the same folder where this README is located)..

After extraction, your folder structure should look exactly like this:
project-folder/
├── README.md
├── your-notebook.ipynb
├── lol-dataset.zip     ← OR     archive.zip  
│
└── ...      



# model.py =============================================
# Features:
- Implements **Zero-DCE++** (Zero-Reference Deep Curve Estimation++) for low-light image enhancement
- Lightweight architecture using depthwise separable convolutions
- U-Net-style encoder-decoder with skip connections
- Predicts spatial-varying curve parameter maps (A) for iterative pixel-wise enhancement
- Outputs both the final enhanced image and the learned parameter maps


# Includes all standard Zero-DCE++ loss functions:
  - Reconstruction loss (L1)
  - Spatial consistency loss
  - Exposure control loss
  - Color constancy loss
  - Illumination smoothness loss
  - SSIM loss (custom implementation)
# Provides utility functions:
  - `spatial_consistency_loss`
  - `exposure_control_loss`
  - `color_constancy_loss`
  - `illumination_smoothness_loss`
  - `ssim` (approximate)
  - `calculate_psnr`
  - `zero_dce_loss` (combined loss with typical weights)

# Requirements:
pip install torch torchvision


# train.py ========================================
# Features:
- Full training pipeline for Zero-DCE++ on the LOL dataset
- Uses AdamW optimizer with Cosine Annealing Warm Restarts scheduler
- Applies gradient clipping for stable training
- Computes and logs training loss every epoch
- Evaluates on test set every 5 epochs using PSNR and SSIM (via torchmetrics)


# Saves:
- Final model based on validation PSNR (`zero_dce_pp_final.pth`)
- Periodic checkpoints every 20 epochs
- Plots training loss curve at the end
- Uses combined Zero-DCE++ loss (reconstruction + spatial + exposure + color + illumination + SSIM)

# Requirements:
pip install torch torchvision torchmetrics tqdm matplotlib numpy



# test.py ========================================
# Features:
- Visualizes Zero-DCE++ enhancement results on the LOL test set
- Full-resolution inference (400×600) — no cropping
- Command-line interface for easy use
- Two modes:
    - Single image: Full-resolution side-by-side + large enhanced view
    - All images: Beautiful vertical grid comparing Input → Ground Truth → Enhanced

- For the model loading, you may either:
  - Uses the provided `zero_dce_pp_trained_default.pth` or;
  - Uses your own `zero_dce_pp_final.pth` trained via `train.py`




# maskloader.py ========================================
# Features:
- Complete pipeline: Low-light enhancement + precise subject segmentation
- Uses **Zero-DCE++** to brighten and restore color in dark images
- Uses **MobileSAM** (fast & lightweight) for high-quality automatic segmentation
- Automatically downloads MobileSAM weights on first run
- Generates a **clean, tight mask** of the main central subject (e.g., the little doll)
- Intelligent scoring + merging + cleanup → removes noise, shadows, and unrelated objects
- Displays clear 3-panel result: Original → Enhanced → Clean Mask

- For the model loading, you may either:
  - Uses the provided `zero_dce_pp_trained_default.pth` or;
  - Uses your own `zero_dce_pp_final.pth` trained via `train.py`

# Requirements
pip install torch torchvision opencv-python matplotlib pillow
pip install git+https://github.com/ChaoningZhang/MobileSAM.git



# masked_spotlight.py ========================================
# Features:
- Creates dramatic spotlight effect on enhanced low-light image
- Uses clean subject mask to:
  - Brighten & boost contrast on the main subject (e.g., the little doll)
  - Darken background for focus and drama
- Option to use pre-generated mask (via `--mask-path`) or auto-generate one (requires `maskloader.py`)
- Full-resolution processing (400×600) for high-quality results

# Requirements
pip install torch torchvision opencv-python matplotlib pillow numpy



# masked_spotlight.py ========================================
# Features:
- Creates dramatic spotlight effect on enhanced low-light image
- Uses clean subject mask to:
  - Brighten & boost contrast on the main subject (e.g., the little doll)
  - Darken background for focus and drama
- Option to use pre-generated mask (via `--mask-path`) or auto-generate one (requires `maskloader.py`)
- Full-resolution processing (400×600) for high-quality results

# Requirements
pip install torch torchvision opencv-python matplotlib pillow numpy
