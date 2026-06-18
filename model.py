import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two Conv3d → InstanceNorm3d → LeakyReLU layers at the same spatial resolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownConv(nn.Module):
    """Learnable stride-2 downsampling (preferred over MaxPool — retains spatial info)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Trilinear upsample (no checkerboard artifacts) → concat skip → ConvBlock."""
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.up      = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.block   = ConvBlock(in_ch, out_ch)
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.dropout(self.block(x))


class UNet3D(nn.Module):
    """
    Custom 3D U-Net with Monte Carlo Dropout for BraTS brain tumor segmentation.

    4 encoder stages + bottleneck + 4 decoder stages.
    Skip connections via concatenation.
    Dropout3d in bottleneck and top-2 decoder blocks.

    Spatial flow (F=32):
      Input  (B,  4, 160, 208, 160)
      enc0   (B, 1F, 160, 208, 160) → skip
      enc1   (B, 2F,  80, 104,  80) → skip
      enc2   (B, 4F,  40,  52,  40) → skip
      enc3   (B, 8F,  20,  26,  20) → skip
      bottle (B, bot,  10,  13,  10)
      dec3   (B, 8F,  20,  26,  20)
      dec2   (B, 4F,  40,  52,  40)
      dec1   (B, 2F,  80, 104,  80)
      dec0   (B, 1F, 160, 208, 160)
      out    (B,  4, 160, 208, 160)  ← raw logits

    Args:
        init_features: base filter count. Use 16 on RTX 3060 (12 GB), 32 on A100.
        dropout_p: MC Dropout probability (keep model.train() at inference for stochasticity).
    """
    def __init__(self, in_channels=4, out_channels=4, init_features=32, dropout_p=0.2):
        super().__init__()
        F   = init_features
        bot = min(F * 16, 320)

        # Encoder
        self.enc0  = ConvBlock(in_channels, F)
        self.down0 = DownConv(F,     F * 2)
        self.enc1  = ConvBlock(F * 2, F * 2)
        self.down1 = DownConv(F * 2, F * 4)
        self.enc2  = ConvBlock(F * 4, F * 4)
        self.down2 = DownConv(F * 4, F * 8)
        self.enc3  = ConvBlock(F * 8, F * 8)
        self.down3 = DownConv(F * 8, bot)

        # Bottleneck — dropout here for MC uncertainty
        self.bottleneck = nn.Sequential(
            nn.Conv3d(bot, bot, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bot, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(bot, bot, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bot, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Decoder — dropout in top-2 blocks (deepest decoder, highest uncertainty benefit)
        self.dec3 = DecoderBlock(bot   + F * 8, F * 8, dropout_p)
        self.dec2 = DecoderBlock(F * 8 + F * 4, F * 4, dropout_p)
        self.dec1 = DecoderBlock(F * 4 + F * 2, F * 2)
        self.dec0 = DecoderBlock(F * 2 + F,     F)

        self.out_conv = nn.Conv3d(F, out_channels, 1)

    def forward(self, x):
        s0 = self.enc0(x)
        x  = self.down0(s0)
        s1 = self.enc1(x)
        x  = self.down1(s1)
        s2 = self.enc2(x)
        x  = self.down2(s2)
        s3 = self.enc3(x)
        x  = self.down3(s3)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.dec0(x, s0)

        return self.out_conv(x)


def mc_inference(model, image, n_passes=10):
    """
    Monte Carlo Dropout inference: N stochastic forward passes with dropout active.

    Uses online mean accumulation to avoid storing N full-volume tensors simultaneously
    (each pass is ~170 MB fp32 for 160x208x160; 20 passes would be ~3.4 GB).

    The returned entropy is H[y|x] = -∑_c p̄_c log(p̄_c), i.e. entropy of the mean
    prediction — this captures total predictive uncertainty (epistemic + aleatoric).

    Args:
        model:    UNet3D — temporarily set to train() to keep dropout active
        image:    (B, 4, H, W, D) input tensor on the correct device
        n_passes: number of stochastic forward passes

    Returns:
        mean_pred: (B, C, H, W, D) softmax probability averaged over passes
        entropy:   (B, H, W, D)    predictive entropy (per-voxel uncertainty)
    """
    prior_training = model.training
    model.train()
    mean_pred = None
    with torch.no_grad():
        for _ in range(n_passes):
            p = torch.softmax(model(image), dim=1)
            mean_pred = p if mean_pred is None else mean_pred + p
    mean_pred = mean_pred / n_passes                                # (B, C, H, W, D)
    entropy   = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(1)  # (B, H, W, D)
    model.train(prior_training)
    return mean_pred, entropy


def get_region_masks(pred_class):
    """
    Derive BraTS evaluation sub-regions from a 4-class argmax map.

    Labels: 0=background, 1=NCR, 2=SNFH/edema, 3=ET

    Returns:
        TC: Tumor Core  = labels 1 + 3
        WT: Whole Tumor = labels 1 + 2 + 3
        ET: Enhancing Tumor = label 3
    """
    TC = (pred_class == 1) | (pred_class == 3)
    WT = (pred_class == 1) | (pred_class == 2) | (pred_class == 3)
    ET = (pred_class == 3)
    return TC.long(), WT.long(), ET.long()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for F in [16, 32]:
        model = UNet3D(init_features=F).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"init_features={F}: {n_params:,} params ({n_params/1e6:.1f}M)")

    model = UNet3D(init_features=16).to(device)
    x = torch.randn(1, 4, 160, 208, 160, device=device)

    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}  (expected: (1, 4, 160, 208, 160))")

    mean_pred, entropy = mc_inference(model, x, n_passes=5)
    print(f"MC mean_pred: {mean_pred.shape}  entropy: {entropy.shape}")
    print(f"Entropy > 0: {(entropy > 0).all().item()}")

    pred_class = mean_pred.argmax(dim=1).squeeze(0)
    TC, WT, ET = get_region_masks(pred_class)
    print(f"TC: {TC.shape}  WT: {WT.shape}  ET: {ET.shape}")