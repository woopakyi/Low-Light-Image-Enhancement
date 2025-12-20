# Low-Light Image Enhancement Project – Split Python Modules

The original `notebook.ipynb` has been refactored into separate, standalone `.py` files for easier reading, inspection, and reuse. Below is a detailed overview of each module.

## dataloader.py
**Features:**
- Extracts the dataset from the zip file
- Locates training (485 pairs) and test (15 pairs) folders
- Applies standard transforms
- Shows sample images

**Requirements:**
```
pip install torch torchvision torchaudio matplotlib tqdm pillow
```

## model.py
**Features:**
- Implements Zero-DCE++ for low-light image enhancement
- Lightweight architecture using depthwise separable convolutions
- U-Net-style encoder-decoder with skip connections
- Predicts spatial-varying curve parameter maps (A) for iterative pixel-wise enhancement
- Outputs both the final enhanced image and the learned parameter maps

**Includes all standard Zero-DCE++ loss functions:**
- Reconstruction loss (L1)
- Spatial consistency loss
- Exposure control loss
- Color constancy loss
- Illumination smoothness loss
- SSIM loss (custom implementation)

**Provides utility functions:**
- `spatial_consistency_loss`
- `exposure_control_loss`
- `color_constancy_loss`
- `illumination_smoothness_loss`
- `ssim` (approximate)
- `calculate_psnr`
- `zero_dce_loss` (combined loss with typical weights)

**Requirements:**
```
pip install torch torchvision
```

## train.py
**Features:**
- Full training pipeline for Zero-DCE++ on the LOL dataset
- Uses AdamW optimizer with Cosine Annealing Warm Restarts scheduler
- Applies gradient clipping for stable training
- Computes and logs training loss every epoch
- Evaluates on test set every 5 epochs using PSNR and SSIM (via torchmetrics)

**Saves:**
- Final model based on validation PSNR (`zero_dce_pp_final.pth`)
- Periodic checkpoints every 20 epochs
- Plots training loss curve at the end
- Uses combined Zero-DCE++ loss (reconstruction + spatial + exposure + color + illumination + SSIM)

**Requirements:**
```
pip install torch torchvision torchmetrics tqdm matplotlib numpy
```

## test.py
**Features:**
- Visualizes Zero-DCE++ enhancement results on the LOL test set
- Full-resolution inference (400×600) — no cropping
- Command-line interface for easy use
- Two modes:
  - Single image: Full-resolution side-by-side + large enhanced view
  - All images: Beautiful vertical grid comparing Input → Ground Truth → Enhanced

- For the model loading, you may either:
  - Use the provided `zero_dce_pp_trained_default.pth` or;
  - Use your own `zero_dce_pp_final.pth` trained via `train.py`

## maskloader.py
**Features:**
- Complete pipeline: Low-light enhancement + precise subject segmentation
- Uses **Zero-DCE++** to brighten and restore color in dark images
- Uses **MobileSAM** (fast & lightweight) for high-quality automatic segmentation
- Automatically downloads MobileSAM weights on first run
- Generates a **clean, tight mask** of the main central subject (e.g., the little doll)
- Intelligent scoring + merging + cleanup → removes noise, shadows, and unrelated objects
- Displays clear 3-panel result: Original → Enhanced → Clean Mask

- For the model loading, you may either:
  - Use the provided `zero_dce_pp_trained_default.pth` or;
  - Use your own `zero_dce_pp_final.pth` trained via `train.py`

**Requirements:**
```
pip install torch torchvision opencv-python matplotlib pillow
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## masked_spotlight.py
**Features:**
- Creates spotlight effect on enhanced low-light image
- Uses clean subject mask to:
  - Brighten & boost contrast on the main subject
  - Darken background for focus and drama
- Full-resolution processing (400×600) for high-quality results

**Requirements:**
```
pip install torch torchvision opencv-python matplotlib pillow numpy
```

## masked_blurry.py
**Features:**
- Creates beautiful bokeh (blurry background) effect on enhanced low-light image
- Uses clean subject mask to:
  - Keep the main subject perfectly sharp
  - Apply strong Gaussian blur to the background for a dreamy, professional portrait look
  - Softly blend the edges with a blurred mask for natural transition
  - Slightly boost the subject's brightness/contrast for extra pop
- Full-resolution processing (400×600) for high-quality results

**Requirements:**
```
pip install torch torchvision opencv-python matplotlib pillow numpy
```