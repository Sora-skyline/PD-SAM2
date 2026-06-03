import math
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

class JointTransform2DVideoTensor:
    def __init__(self,
                 img_size=256,
                 crop=None,                 # (h,w) or None
                 p_flip=0.0,
                 p_rota=0.0,                # rotate only
                 p_scale=0.0,               # scale + random crop back
                 p_gaussn=0.0,              # gaussian noise
                 p_contr=0.0,               # extra contrast jitter (separate from color jitter)
                 p_gama=0.0,                # gamma
                 p_distor=0.0,              # shear
                 color_jitter_params=None,  # (b,c,s,h) ranges
                 p_random_affine=0.0,
                 affine_cfg=None,           # dict override for affine ranges
                 ):
        self.img_size = img_size
        self.crop = crop

        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_contr = p_contr
        self.p_gama = p_gama
        self.p_distor = p_distor
        self.p_random_affine = p_random_affine

        self.bj, self.cj, self.sj, self.hj = color_jitter_params or (0,0,0,0)

        # 默认把 affine 范围调“温和一些”，更适合医学超声
        self.affine_cfg = affine_cfg or {
            "angle": (-30.0, 30.0),
            "scale": (1, 1.3),
            "shear": (-10.0, 10.0),  # 这里用统一 shear_x / shear_y
            "translate": (0.0, 0.0), # fraction, 暂时不用平移
        }

    @staticmethod
    def _rand01(device):
        return torch.rand((), device=device)

    @staticmethod
    def _to_float01(img: torch.Tensor) -> torch.Tensor:
        # img: (F,C,H,W), uint8 or float
        if img.dtype == torch.uint8:
            return img.float() / 255.0
        img = img.float()
        # 若看起来像 0..255，则归一化
        if img.max() > 1.5:
            img = img / 255.0
        return img.clamp(0.0, 1.0)

    @staticmethod
    def _ensure_mask(mask: torch.Tensor) -> torch.Tensor:
        # mask: (F,H,W) or (F,1,H,W) -> (F,1,H,W) float
        if mask.ndim == 4 and mask.shape[1] == 1:
            m = mask
        elif mask.ndim == 3:
            m = mask[:, None, :, :]
        else:
            raise ValueError(f"Expected mask (F,H,W) or (F,1,H,W), got {tuple(mask.shape)}")
        m = m.float()
        # 容忍输入是 0/255 或 0/1
        if m.max() > 1.5:
            m = (m > 127.5).float()
        else:
            m = (m > 0.5).float()
        return m

    @staticmethod
    def _crop(img, top, left, h, w):
        return img[..., top:top+h, left:left+w]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        """
        image: (F,C,H,W) torch tensor
        mask : (F,H,W) or (F,1,H,W)
        return:
          image_t: (F,C,img_size,img_size) float32 [0,1]
          mask_t : (F,img_size,img_size) float32 {0,1}
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image (F,C,H,W), got {tuple(image.shape)}")

        device = image.device
        F_, C, H, W = image.shape

        img = self._to_float01(image)
        msk = self._ensure_mask(mask).to(device=device)

        # ---------- gamma (sync) ----------
        if self._rand01(device) < self.p_gama:
            g = torch.empty((), device=device).uniform_(1.0, 2.5)  # ~[1.0,2.5]
            # 你的原式是 (img^(1/g))，这里保持一致
            img = img.clamp(1e-6, 1.0).pow(1.0 / g)

        # ---------- random crop (sync) ----------
        if self.crop is not None:
            ch, cw = self.crop
            if ch > H or cw > W:
                # 如果 crop 比原图大，先 resize 到至少 crop 大小
                newH = max(H, ch)
                newW = max(W, cw)
                img = F.interpolate(img, size=(newH, newW), mode="bilinear", align_corners=False)
                msk = F.interpolate(msk, size=(newH, newW), mode="nearest")
                H, W = newH, newW

            top  = torch.randint(0, H - ch + 1, (1,), device=device).item()
            left = torch.randint(0, W - cw + 1, (1,), device=device).item()
            img = self._crop(img, top, left, ch, cw)
            msk = self._crop(msk, top, left, ch, cw)
            H, W = ch, cw

        # ---------- hflip (sync) ----------
        if self._rand01(device) < self.p_flip:
            img = torch.flip(img, dims=[-1])
            msk = torch.flip(msk, dims=[-1])

        # ---------- rotate (sync) ----------
        if self._rand01(device) < self.p_rota:
            angle = float(torch.empty((), device=device).uniform_(-30.0, 30.0).item())
            img = TF.rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR)
            msk = TF.rotate(msk, angle=angle, interpolation=InterpolationMode.NEAREST)

        # ---------- scale + crop back (sync) ----------
        if self._rand01(device) < self.p_scale:
            scale = float(torch.empty((), device=device).uniform_(1.0, 1.3).item())
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
            img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
            msk = F.interpolate(msk, size=(new_h, new_w), mode="nearest")

            # crop back to original H,W (or img_size if你希望固定)
            target_h, target_w = H, W
            if new_h > target_h and new_w > target_w:
                top  = torch.randint(0, new_h - target_h + 1, (1,), device=device).item()
                left = torch.randint(0, new_w - target_w + 1, (1,), device=device).item()
                img = self._crop(img, top, left, target_h, target_w)
                msk = self._crop(msk, top, left, target_h, target_w)

        # ---------- shear/distortion (sync) ----------
        if self._rand01(device) < self.p_distor:
            shear_x = float(torch.empty((), device=device).uniform_(5.0, 30.0).item())
            shear = [shear_x, 0.0]
            img = TF.affine(
                img, angle=0.0, translate=[0, 0], scale=1.0, shear=shear,
                interpolation=InterpolationMode.BILINEAR
            )
            msk = TF.affine(
                msk, angle=0.0, translate=[0, 0], scale=1.0, shear=shear,
                interpolation=InterpolationMode.NEAREST
            )

        # ---------- random affine (sync, moderate ranges) ----------
        if self._rand01(device) < self.p_random_affine:
            ang0, ang1 = self.affine_cfg["angle"]
            sc0, sc1   = self.affine_cfg["scale"]
            sh0, sh1   = self.affine_cfg["shear"]

            angle = float(torch.empty((), device=device).uniform_(ang0, ang1).item())
            scale = float(torch.empty((), device=device).uniform_(sc0, sc1).item())
            shear_x = float(torch.empty((), device=device).uniform_(sh0, sh1).item())
            shear_y = float(torch.empty((), device=device).uniform_(sh0, sh1).item())
            shear = [shear_x, shear_y]

            img = TF.affine(
                img, angle=angle, translate=[0, 0], scale=scale, shear=shear,
                interpolation=InterpolationMode.BILINEAR
            )
            msk = TF.affine(
                msk, angle=angle, translate=[0, 0], scale=scale, shear=shear,
                interpolation=InterpolationMode.NEAREST
            )

        # ---------- gaussian noise (sync strength, but per-pixel) ----------
        if self._rand01(device) < self.p_gaussn:
            sigma = float(torch.empty((), device=device).uniform_(3.0/255.0, 15.0/255.0).item())
            img = (img + torch.randn_like(img) * sigma).clamp(0.0, 1.0)

        # ---------- extra contrast jitter (sync) ----------
        if self._rand01(device) < self.p_contr:
            c = float(torch.empty((), device=device).uniform_(0.8, 2.0).item())
            img = TF.adjust_contrast(img, c).clamp(0.0, 1.0)

        # ---------- ColorJitter (sync params across frames) ----------
        # brightness/contrast/saturation in [1-b, 1+b]; hue in [-h, h]
        if any(v > 0 for v in (self.bj, self.cj, self.sj, self.hj)):
            if self.bj > 0:
                b = float(torch.empty((), device=device).uniform_(max(0.0, 1.0 - self.bj), 1.0 + self.bj).item())
                img = TF.adjust_brightness(img, b)
            if self.cj > 0:
                c = float(torch.empty((), device=device).uniform_(max(0.0, 1.0 - self.cj), 1.0 + self.cj).item())
                img = TF.adjust_contrast(img, c)
            if self.sj > 0:
                s = float(torch.empty((), device=device).uniform_(max(0.0, 1.0 - self.sj), 1.0 + self.sj).item())
                img = TF.adjust_saturation(img, s)
            if self.hj > 0:
                h = float(torch.empty((), device=device).uniform_(-self.hj, self.hj).item())
                img = TF.adjust_hue(img, h)
            img = img.clamp(0.0, 1.0)

        # ---------- final resize ----------
        img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        msk = F.interpolate(msk, size=(self.img_size, self.img_size), mode="nearest")

        # ---------- final mask format ----------
        mask_out = (msk > 0.5).float().squeeze(1)  # (F,H,W)
        return img.contiguous(), mask_out.contiguous()
