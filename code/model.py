import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthWiseConv(nn.Module):
    """Depthwise separable convolution block with ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.point = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        return self.relu(x)


class ZeroDCEPlusPlus(nn.Module):
    """
    Zero-DCE++ model for low-light image enhancement.
    Outputs enhanced image and the learned curve parameter maps.
    """
    def __init__(self, num_iter: int = 8):
        super().__init__()
        self.num_iter = num_iter

        # Encoder
        self.conv1 = DepthWiseConv(3,   32)
        self.conv2 = DepthWiseConv(32,  64)
        self.conv3 = DepthWiseConv(64,  128)
        self.conv4 = DepthWiseConv(128, 256)
        self.conv5 = DepthWiseConv(256, 512)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (upsampling with skip connections)
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)

        # Final head to predict enhancement curves
        self.head = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 64 (d3) + 64 (x2)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3 * num_iter, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        p2 = self.pool(x2)

        x3 = self.conv3(p2)
        p3 = self.pool(x3)

        x4 = self.conv4(p3)
        p4 = self.pool(x4)

        x5 = self.conv5(p4)  # Bottleneck

        # Decoder with skip connections
        d5 = self.up5(x5) + x4
        d4 = self.up4(d5) + x3
        d3 = self.up3(d4) + x2

        # Predict enhancement parameter maps
        params = self.head(torch.cat([d3, x2], dim=1))  # (B, 3*num_iter, H, W)
        B, _, H, W = params.shape
        params = params.view(B, self.num_iter, 3, H, W)
        params = params * 5.0  # Scale to larger range as in Zero-DCE++

        # Iterative enhancement
        enhanced = x
        for i in range(self.num_iter):
            A = params[:, i]  # (B, 3, H, W)
            enhanced = enhanced + A * (torch.pow(enhanced, 2) - enhanced)
            enhanced = torch.clamp(enhanced, 0.0, 1.0)

        return enhanced, params


# ============================
# Loss Functions
# ============================

def spatial_consistency_loss(enhanced: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    """Encourage spatial consistency between input and enhanced image."""
    def gradient(x):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    e_gray = enhanced.mean(1, keepdim=True)
    l_gray = low.mean(1, keepdim=True)

    e_dx, e_dy = gradient(e_gray)
    l_dx, l_dy = gradient(l_gray)

    w_x = torch.exp(-torch.mean(torch.abs(l_dx), dim=1, keepdim=True))
    w_y = torch.exp(-torch.mean(torch.abs(l_dy), dim=1, keepdim=True))

    loss_x = torch.mean(w_x * torch.abs(e_dx))
    loss_y = torch.mean(w_y * torch.abs(e_dy))
    return loss_x + loss_y


def exposure_control_loss(enhanced: torch.Tensor, target_exposure: float = 0.6) -> torch.Tensor:
    """Penalize over/under exposure."""
    enhanced_gray = enhanced.mean(1, keepdim=True)
    return torch.mean((enhanced_gray - target_exposure) ** 2)


def color_constancy_loss(enhanced: torch.Tensor) -> torch.Tensor:
    """Simple color balance loss."""
    r, g, b = torch.split(enhanced, 1, dim=1)
    mean_r = r.mean(dim=[2, 3])
    mean_g = g.mean(dim=[2, 3])
    mean_b = b.mean(dim=[2, 3])
    return torch.mean((mean_r - mean_g)**2 + (mean_g - mean_b)**2 + (mean_b - mean_r)**2)


def illumination_smoothness_loss(params: torch.Tensor) -> torch.Tensor:
    """Smoothness on illumination maps (A)."""
    loss = 0.0
    for i in range(params.size(1)):
        A = params[:, i]  # (B, 3, H, W)
        loss += torch.mean(torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:]))   # horizontal
        loss += torch.mean(torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :]))   # vertical
    return loss / params.size(1)


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Simple SSIM implementation (approximate, without gaussian weighting)."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


# Optional: Combined loss (common weights used in papers)
def zero_dce_loss(enhanced, high, params, low, ssim_fn=None):
    recon_loss = nn.L1Loss()(enhanced, high)

    s_loss = spatial_consistency_loss(enhanced, low)
    e_loss = exposure_control_loss(enhanced)
    c_loss = color_constancy_loss(enhanced)
    i_loss = illumination_smoothness_loss(params)

    if ssim_fn is None:
        ssim_val = ssim(enhanced, high)
    else:
        ssim_val = ssim_fn(enhanced, high)

    total = (recon_loss +
             10.0 * s_loss +
             10.0 * e_loss +
             10.0 * c_loss +
             0.1 * i_loss +
             5.0 * (1 - ssim_val))

    return total