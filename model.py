import timm
import torch
import torch.nn as nn


class FrequencyAwareLoss(nn.Module):
    # Classification Loss (CE) with a Fourier Frequency Loss (FT)
    def __init__(self, ce_weight=1.0, ft_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.ft_weight = ft_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels, images):
        # Standard Cross Entropy Loss
        base_loss = self.ce(logits, labels)
        # batch to grayscal and float32
        gray = images.mean(dim=1, keepdim=True).float()
        # 2D Fast Fourier Transorm
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.log(torch.abs(fft_shift) + 1e-8)

        # Create High Pass Filter Mask
        B, C, H, W = magnitude.shape
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        Y, X = Y.to(images.device), X.to(images.device)
        center_y, center_x = H // 2, W // 2

        radius = 25
        dist = (X - center_x) ** 2 + (Y - center_y) ** 2
        mask = (dist >= radius**2).float().unsqueeze(0).unsqueeze(0)

        # High frequency energy per image
        hf_energy = (magnitude * mask).mean(dim=(1, 2, 3))

        # Normalize HF energy to [0, 1] range across the batch
        hf_min = hf_energy.min()
        hf_max = hf_energy.max()
        if hf_max > hf_min:
            hf_normalized = (hf_energy - hf_min) / (hf_max - hf_min + 1e-8)
        else:
            hf_normalized = torch.zeros_like(hf_energy)

        # Fourier Auxiliary LOSS (MSE)
        probs = torch.softmax(logits, dim=1)
        spoof_probs = probs[:, 1]  # Spoof
        ft_loss = nn.functional.mse_loss(spoof_probs, hf_normalized.detach())

        return (self.ce_weight * base_loss) + (self.ft_weight * ft_loss)


num_classes = 2
model_name = "swin_tiny_patch4_window7_224"

# Replace ImageNet Cl
model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

print("Model Architecture: Swin-Tiny Transformer")
print("Classification head: ", model.head)
