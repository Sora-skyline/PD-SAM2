import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """
    t: [B] float/int
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=t.device).float() / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def make_gn(num_channels, max_groups=8):
    for g in [max_groups, 4, 2, 1]:
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)


# =========================
# basic blocks
# =========================
class ConvGnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.norm = make_gn(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlockFiLM(nn.Module):
    """
    保守稳定版：
    - 单次 FiLM 注入
    - 无 dropout
    - GN + SiLU
    """
    def __init__(self, in_ch, out_ch, temb_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.norm1 = make_gn(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.norm2 = make_gn(out_ch)

        self.act = nn.SiLU(inplace=True)

        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(temb_dim, out_ch * 2)
        )

        # zero init: 稳定
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb):
        residual = self.shortcut(x)

        h = self.conv1(x)
        h = self.norm1(h)

        ss = self.to_scale_shift(temb)
        scale, shift = ss.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = h * (1 + scale) + shift

        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return self.act(h + residual)


# =========================
# SAM adapter
# =========================
class SAM2Adapter(nn.Module):
    """
    输入 SAM feats:
      feat0: [B,  32, 64, 64]
      feat1: [B,  64, 32, 32]
      feat2: [B, 256, 16, 16]

    额外生成:
      level4: [B, 256, 8, 8]
    """
    def __init__(self, in_ch_list=(32, 64, 256)):
        super().__init__()
        c3 = in_ch_list[2]
        self.make_level4 = nn.Sequential(
            nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1, bias=False),
            make_gn(c3, max_groups=32),
            nn.SiLU(inplace=True),
        )

    def forward(self, feats_list):
        assert len(feats_list) == 3, f"Expect 3 SAM feature maps, got {len(feats_list)}"
        f0, f1, f2 = feats_list
        f3 = self.make_level4(f2)

        return {
            "level1": f0,  # 64x64
            "level2": f1,  # 32x32
            "level3": f2,  # 16x16
            "level4": f3,  # 8x8
        }


# =========================
# simple fusion blocks
# =========================
class FeatureFusion3(nn.Module):
    """
    image + mask + sam -> out
    保守版：简单 concat，不加 gate
    """
    def __init__(self, image_channels, mask_channels, sam_channels, out_channels):
        super().__init__()
        self.image_conv = ConvGnAct(image_channels, out_channels)
        self.mask_conv  = ConvGnAct(mask_channels,  out_channels)
        self.sam_conv   = ConvGnAct(sam_channels,   out_channels)
        self.out_conv   = ConvGnAct(out_channels * 3, out_channels)

    def forward(self, image_feat, mask_feat, sam_feat):
        a = self.image_conv(image_feat)
        b = self.mask_conv(mask_feat)
        s = self.sam_conv(sam_feat)

        if s.shape[-2:] != a.shape[-2:]:
            s = F.interpolate(s, size=a.shape[-2:], mode='bilinear', align_corners=False)

        return self.out_conv(torch.cat([a, b, s], dim=1))


class FeatureFusion2(nn.Module):
    """
    x + mask + sam -> out
    """
    def __init__(self, x_channels, mask_channels, sam_channels, out_channels):
        super().__init__()
        self.x_conv    = ConvGnAct(x_channels, out_channels)
        self.mask_conv = ConvGnAct(mask_channels, out_channels)
        self.sam_conv  = ConvGnAct(sam_channels, out_channels)
        self.out_conv  = ConvGnAct(out_channels * 3, out_channels)

    def forward(self, x_feat, mask_feat, sam_feat):
        a = self.x_conv(x_feat)
        b = self.mask_conv(mask_feat)
        s = self.sam_conv(sam_feat)

        if b.shape[-2:] != a.shape[-2:]:
            b = F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)
        if s.shape[-2:] != a.shape[-2:]:
            s = F.interpolate(s, size=a.shape[-2:], mode='bilinear', align_corners=False)

        return self.out_conv(torch.cat([a, b, s], dim=1))


# =========================
# refine step
# =========================
class IterativeRefinerStep(nn.Module):
    """
    输入:
      images:      [B, 3, 256, 256]
      mask_logits: [B, 1, 256, 256]
      sam_feats:
        level1: [B,  32, 64, 64]
        level2: [B,  64, 32, 32]
        level3: [B, 256, 16, 16]
        level4: [B, 256,  8,  8]

    输出:
      delta_logits: [B, 1, 256, 256]
    """
    def __init__(self, base_c=16, temb_dim=64):
        super().__init__()
        self.temb_dim = temb_dim
        self.base_c = base_c

        # ---- time embedding ----
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim * 4),
            nn.SiLU(),
            nn.Linear(temb_dim * 4, temb_dim),
        )

        # ---- stems: 256 -> 128 -> 64 ----
        self.img_stem = nn.Sequential(
            ConvGnAct(3, base_c, kernel_size=3, stride=2, padding=1),      # 256 -> 128
            ConvGnAct(base_c, base_c, kernel_size=3, stride=2, padding=1), # 128 -> 64
        )

        # mask input: logits + prob = 2 channels
        self.mask_stem = nn.Sequential(
            ConvGnAct(2, base_c, kernel_size=3, stride=2, padding=1),      # 256 -> 128
            ConvGnAct(base_c, base_c, kernel_size=3, stride=2, padding=1), # 128 -> 64
        )

        # mask pyramid: 64 -> 32 -> 16 -> 8
        self.mask_down1 = ConvGnAct(base_c,   base_c,   stride=2)  # 64 -> 32
        self.mask_down2 = ConvGnAct(base_c,   base_c*2, stride=2)  # 32 -> 16
        self.mask_down3 = ConvGnAct(base_c*2, base_c*4, stride=2)  # 16 -> 8

        # ---- encoder ----
        self.fuse1 = FeatureFusion3(
            image_channels=base_c,
            mask_channels=base_c,
            sam_channels=32,
            out_channels=base_c
        )
        self.enc1 = ResBlockFiLM(base_c, base_c, temb_dim)

        self.fuse2 = FeatureFusion2(
            x_channels=base_c,
            mask_channels=base_c,
            sam_channels=64,
            out_channels=base_c * 2
        )
        self.enc2 = ResBlockFiLM(base_c * 2, base_c * 2, temb_dim)

        self.fuse3 = FeatureFusion2(
            x_channels=base_c * 2,
            mask_channels=base_c * 2,
            sam_channels=256,
            out_channels=base_c * 4
        )
        self.enc3 = ResBlockFiLM(base_c * 4, base_c * 4, temb_dim)

        self.fuse4 = FeatureFusion2(
            x_channels=base_c * 4,
            mask_channels=base_c * 4,
            sam_channels=256,
            out_channels=base_c * 8
        )
        self.enc4 = ResBlockFiLM(base_c * 8, base_c * 8, temb_dim)

        # ---- bottleneck ----
        # 注意：这里保留额外 pool，避免 decoder 第一层 skip 错位
        self.bot = ResBlockFiLM(base_c * 8, base_c * 8, temb_dim)

        # ---- decoder ----
        self.up4  = nn.ConvTranspose2d(base_c * 8, base_c * 4, 2, 2)
        self.dec4 = ResBlockFiLM(base_c * 4 + base_c * 8, base_c * 4, temb_dim)

        self.up3  = nn.ConvTranspose2d(base_c * 4, base_c * 2, 2, 2)
        self.dec3 = ResBlockFiLM(base_c * 2 + base_c * 4, base_c * 2, temb_dim)

        self.up2  = nn.ConvTranspose2d(base_c * 2, base_c, 2, 2)
        self.dec2 = ResBlockFiLM(base_c + base_c * 2, base_c, temb_dim)

        self.up1  = nn.ConvTranspose2d(base_c, base_c, 2, 2)
        self.dec1 = ResBlockFiLM(base_c + base_c, base_c, temb_dim)

        self.delta_head = nn.Conv2d(base_c, 1, 1)

        # 小步长，减少震荡
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def _make_mask_features(self, mask_logits):
        prob = torch.sigmoid(mask_logits)
        mask_input = torch.cat([mask_logits, prob], dim=1)  # [B, 2, 256, 256]

        m1 = self.mask_stem(mask_input)  # [B, C, 64, 64]
        m2 = self.mask_down1(m1)         # [B, C, 32, 32]
        m3 = self.mask_down2(m2)         # [B, 2C, 16, 16]
        m4 = self.mask_down3(m3)         # [B, 4C, 8, 8]
        return m1, m2, m3, m4

    def forward(self, images, mask_logits, sam_feats, t, detach_mask=False):
        if detach_mask:
            mask_logits = mask_logits.detach()

        # ---- time embedding ----
        temb = timestep_embedding(t, self.temb_dim)
        temb = self.time_mlp(temb)

        # ---- image / mask pyramid ----
        x_img = self.img_stem(images)   # [B, C, 64, 64]
        m1, m2, m3, m4 = self._make_mask_features(mask_logits)

        # ---- shape checks ----
        assert x_img.shape[-2:] == (64, 64), f"x_img shape should be 64x64, got {x_img.shape}"
        assert sam_feats["level1"].shape[-2:] == (64, 64), f"level1 mismatch: {sam_feats['level1'].shape}"
        assert sam_feats["level2"].shape[-2:] == (32, 32), f"level2 mismatch: {sam_feats['level2'].shape}"
        assert sam_feats["level3"].shape[-2:] == (16, 16), f"level3 mismatch: {sam_feats['level3'].shape}"
        assert sam_feats["level4"].shape[-2:] == (8, 8), f"level4 mismatch: {sam_feats['level4'].shape}"

        # =========================
        # encoder
        # =========================
        # 64x64
        f1 = self.fuse1(x_img, m1, sam_feats["level1"])
        e1 = self.enc1(f1, temb)
        x = F.max_pool2d(e1, 2)  # -> 32

        # 32x32
        f2 = self.fuse2(x, m2, sam_feats["level2"])
        e2 = self.enc2(f2, temb)
        x = F.max_pool2d(e2, 2)  # -> 16

        # 16x16
        f3 = self.fuse3(x, m3, sam_feats["level3"])
        e3 = self.enc3(f3, temb)
        x = F.max_pool2d(e3, 2)  # -> 8

        # 8x8
        f4 = self.fuse4(x, m4, sam_feats["level4"])
        e4 = self.enc4(f4, temb)

        # =========================
        # bottleneck
        # =========================
        x = F.max_pool2d(e4, 2)  # 8 -> 4
        x = self.bot(x, temb)    # 4x4

        # =========================
        # decoder
        # =========================
        x = self.up4(x)          # 4 -> 8
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x, temb)

        x = self.up3(x)          # 8 -> 16
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x, temb)

        x = self.up2(x)          # 16 -> 32
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x, temb)

        x = self.up1(x)          # 32 -> 64
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x, temb)

        # 64 -> 256
        delta_logits = self.delta_head(x)
        delta_logits = F.interpolate(
            delta_logits,
            size=mask_logits.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        delta_logits = self.delta_scale * delta_logits
        return delta_logits


# =========================
# outer wrapper
# =========================
class IterativeLogitsRefiner(nn.Module):
    def __init__(self, K=1, base_c=16, temb_dim=64):
        super().__init__()
        self.K_train = K

        self.step = IterativeRefinerStep(
            base_c=base_c,
            temb_dim=temb_dim
        )
        self.sam_adapter = SAM2Adapter(in_ch_list=(32, 64, 256))

    def forward(
        self,
        images,
        coarse_logits,
        sam_feats_list,
        detach_coarse=False,
        return_all=False,
        inference_K=None,
    ):
        """
        images: [B, 3, 256, 256]
        coarse_logits: [B, 1, 256, 256]
        sam_feats_list:
            [
              [B,  32, 64, 64],
              [B,  64, 32, 32],
              [B, 256, 16, 16],
            ]
        """
        m = coarse_logits.detach() if detach_coarse else coarse_logits
        K = self.K_train if inference_K is None else inference_K

        sam_feats = self.sam_adapter(sam_feats_list)
        outs = []

        denom = max(K - 1, 1)

        for i in range(K):
            t = torch.full(
                (images.size(0),),
                fill_value=float(i) / denom,
                device=images.device,
                dtype=torch.float32
            )
            delta = self.step(images, m, sam_feats, t)
            m = m + delta
            outs.append(m)

        if return_all:
            return outs, m
        return m