import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Import modules
from model import ZeroDCEPlusPlus, zero_dce_loss
from dataloader import train_loader, test_loader, device  # device should be defined in dataloader.py


# ==============================
# Metrics (using torchmetrics for accuracy)
# ==============================
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # for loss weighting


def calculate_psnr(img1, img2):
    return psnr_metric(img1, img2)


# ==============================
# Model, Optimizer, Scheduler
# ==============================
model = ZeroDCEPlusPlus(num_iter=8).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-6
)

num_epochs = 200  # Common setting for Zero-DCE++ convergence
best_psnr = 0.0

# Logging
train_losses = []
val_psnrs = []
val_ssims = []


# ==============================
# Training Loop
# ==============================
print("Starting training...\n")
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{num_epochs}")

    for low, high in progress_bar:
        low, high = low.to(device), high.to(device)

        optimizer.zero_grad()
        enhanced, params = model(low)
        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        loss = zero_dce_loss(enhanced, high, params, low, ssim_fn=ssim_loss_fn)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.5f}"})

    scheduler.step()
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | LR: {current_lr:.6f}")

    # Validation every 5 epochs
    if epoch % 5 == 0 or epoch == num_epochs:
        model.eval()
        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for low, high in test_loader:
                low, high = low.to(device), high.to(device)
                enhanced, _ = model(low)
                enhanced = torch.clamp(enhanced, 0.0, 1.0)

                psnr_list.append(calculate_psnr(enhanced, high).item())
                ssim_list.append(ssim_metric(enhanced, high).item())

        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)

        print(f"Validation → PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'best_zero_dce_pp.pth')
            print(f"  >>> NEW BEST MODEL SAVED! PSNR: {best_psnr:.2f} dB")

        # Periodic checkpoint
        if epoch % 20 == 0 or epoch == num_epochs:
            torch.save(model.state_dict(), f'zero_dce_pp_epoch_{epoch}.pth')
            print(f"  >>> Checkpoint saved: zero_dce_pp_epoch_{epoch}.pth")

        print("-" * 70)


# ==============================
# Final Results & Plot
# ==============================
print(f"\nTraining completed!")
print(f"Best Validation PSNR: {best_psnr:.2f} dB")
print(f"Final model saved as 'best_zero_dce_pp.pth'\n")

# Plot training loss
plt.figure(figsize=(12, 6))
epochs = np.arange(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Training Loss', color='tab:blue', linewidth=2)
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()