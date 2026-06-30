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
    """Learnable stride-2 downsampling (retains spatial info vs MaxPool)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Soft attention gate from Oktay et al. 2018 (Attention U-Net).

    Projects g (decoder gating signal) and x (skip features) to F_int channels,
    adds them, applies sigmoid to produce a per-voxel attention map, then
    returns x scaled by that map — suppressing irrelevant encoder activations.

    g  : upsampled decoder features (same spatial size as x after upsampling)
    x  : encoder skip features to be filtered
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g,  F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l,  F_int, kernel_size=1, bias=True),
            nn.InstanceNorm3d(F_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """Trilinear upsample → attention-gate skip → concat → ConvBlock.

    Args:
        g_ch   : channels of the incoming decoder tensor (gating signal)
        skip_ch: channels of the encoder skip connection to be filtered
        out_ch : output channels after ConvBlock
    """
    def __init__(self, g_ch, skip_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.up      = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.gate    = AttentionGate(F_g=g_ch, F_l=skip_ch, F_int=max(skip_ch // 2, 8))
        self.block   = ConvBlock(g_ch + skip_ch, out_ch)
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x, skip):
        x    = self.up(x)
        skip = self.gate(g=x, x=skip)
        x    = torch.cat([x, skip], dim=1)
        return self.dropout(self.block(x))


class UNet3DAttn(nn.Module):
    """
    3D Attention U-Net with MC Dropout for BraTS brain tumor segmentation.

    Identical encoder/decoder topology to UNet3D but with attention-gated
    skip connections (Oktay et al. 2018). Each gate uses the decoder signal
    to suppress irrelevant encoder activations — particularly beneficial for
    small regions (ET, NCR).

    Channel flow (F=32, bot=min(16F,320)=320):
      Input  (B,  4, 160, 208, 160)
      enc0   (B, 32, 160, 208, 160) → skip s0
      enc1   (B, 64,  80, 104,  80) → skip s1
      enc2   (B,128,  40,  52,  40) → skip s2
      enc3   (B,256,  20,  26,  20) → skip s3
      bottle (B,320,  10,  13,  10)
      dec3   gate(320,256)→cat(320+256)→256
      dec2   gate(256,128)→cat(256+128)→128
      dec1   gate(128, 64)→cat(128+ 64)→ 64
      dec0   gate( 64, 32)→cat( 64+ 32)→ 32
      out    (B,  4, 160, 208, 160)
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

        # Bottleneck with dropout for MC uncertainty
        self.bottleneck = nn.Sequential(
            nn.Conv3d(bot, bot, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bot, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(bot, bot, 3, padding=1, bias=False),
            nn.InstanceNorm3d(bot, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

        # Decoder with attention gates; dropout in top-2 (deepest) decoder blocks
        self.dec3 = DecoderBlock(bot,   F * 8, F * 8, dropout_p)
        self.dec2 = DecoderBlock(F * 8, F * 4, F * 4, dropout_p)
        self.dec1 = DecoderBlock(F * 4, F * 2, F * 2)
        self.dec0 = DecoderBlock(F * 2, F,     F)

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

    Uses online mean accumulation to avoid storing N full-volume tensors simultaneously.
    Returned entropy is H[y|x] = -∑_c p̄_c log(p̄_c) (entropy of the mean prediction).

    Args:
        model:    UNet3DAttn — kept in train() mode so dropout stays active
        image:    (B, 4, H, W, D) input tensor on the correct device
        n_passes: number of stochastic forward passes

    Returns:
        mean_pred: (B, C, H, W, D) softmax probability averaged over passes
        entropy:   (B, H, W, D)    per-voxel predictive uncertainty
    """
    prior_training = model.training
    model.train()
    mean_pred = None
    with torch.no_grad():
        for _ in range(n_passes):
            p = torch.softmax(model(image), dim=1)
            mean_pred = p if mean_pred is None else mean_pred + p
    mean_pred = mean_pred / n_passes
    entropy   = -(mean_pred * torch.log(mean_pred + 1e-8)).sum(1)
    model.train(prior_training)
    return mean_pred, entropy


def get_region_masks(pred_class):
    """
    Derive BraTS evaluation sub-regions from a 4-class argmax map.
    Labels: 0=background, 1=NCR, 2=SNFH/edema, 3=ET
    TC = 1+3 | WT = 1+2+3 | ET = 3
    """
    TC = (pred_class == 1) | (pred_class == 3)
    WT = (pred_class == 1) | (pred_class == 2) | (pred_class == 3)
    ET = (pred_class == 3)
    return TC.long(), WT.long(), ET.long()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for F in [16, 32]:
        model = UNet3DAttn(init_features=F).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"UNet3DAttn init_features={F}: {n_params:,} params ({n_params/1e6:.1f}M)")

    model = UNet3DAttn(init_features=16).to(device)
    x = torch.randn(1, 4, 160, 208, 160, device=device)
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")

    mean_pred, entropy = mc_inference(model, x, n_passes=5)
    print(f"MC mean_pred: {mean_pred.shape}  entropy: {entropy.shape}")

    pred_class = mean_pred.argmax(dim=1).squeeze(0)
    TC, WT, ET = get_region_masks(pred_class)
    print(f"TC: {TC.shape}  WT: {WT.shape}  ET: {ET.shape}")