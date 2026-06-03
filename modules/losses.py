from typing import Optional, Dict, Any, Iterable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    # def _dice_loss(self, score, target):
    #     target = target.float()
    #     smooth = 1e-5
    #     dims = (1, 2, 3)  # [N, C, H, W] -> 针对每张图计算
        
    #     intersect = torch.sum(score * target, dims)
        
    #     y_sum = torch.sum(target, dims) 
    #     z_sum = torch.sum(score, dims)
        
    #     dice = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        
    #     return (1 - dice).mean()

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output)
        
        # 兼容 B,1,H,W 和 B,T,1,H,W 等情况，只要最后尺寸匹配
        # 这里假设输入已经调整好维度或者是标准的分割输出
        if net_output.shape != target.shape:
             # 简单的防错机制，实际应在外部保证维度一致
             pass 

        # 对于单通道二分类，取第0通道
        dice_loss = self._dice_loss(net_output, target)
        return dice_loss

class EdgeLoss(nn.Module):

    """BoundaryDoU-style edge loss with binary and multi-class support."""

    def __init__(
        self,
        classes: Optional[int] = None,
        alpha_trunc: float = 0.8,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.classes = classes
        self.alpha_trunc = alpha_trunc
        self.smooth = smooth
        self.register_buffer(
            "kernel",
            torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32),
        )

    def _one_hot(self, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        if target.dim() == 4 and target.size(1) == num_classes:
            return (target > 0.5).float()
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        if target.dim() != 3:
            raise ValueError(f"Unsupported target shape for one-hot encoding: {target.shape}")
        target = target.long()
        return F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

    def _boundary_dou(self, score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.to(device=score.device, dtype=score.dtype).view(1, 1, 3, 3)
        target_4d = target.unsqueeze(1)
        conv_map = F.conv2d(target_4d, kernel, padding=1)
        boundary_map = conv_map.squeeze(1) * target
        boundary_map = torch.where(boundary_map == 5, torch.zeros_like(boundary_map), boundary_map)

        c_num = torch.count_nonzero(boundary_map)
        s_num = torch.count_nonzero(target)
        alpha = 1.0 - (c_num + self.smooth) / (s_num + self.smooth)
        alpha = 2.0 * alpha - 1.0
        alpha = torch.clamp(alpha, max=self.alpha_trunc)

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (z_sum + y_sum - 2.0 * intersect + self.smooth) / (
            z_sum + y_sum - (1.0 + alpha) * intersect + self.smooth
        )
        return loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 5:
            logits = logits.view(-1, logits.size(2), logits.size(3), logits.size(4))
            target = target.view(-1, target.size(2), target.size(3), target.size(4))

        num_classes = logits.shape[1] if self.classes is None else self.classes
        if num_classes != logits.shape[1]:
            num_classes = logits.shape[1]

        if num_classes == 1:
            prob = torch.sigmoid(logits)
            if target.dim() == prob.dim() - 1:
                target = target.unsqueeze(1)
            target_bin = (target > 0.5).float()
            score = prob.squeeze(1)
            target_plane = target_bin.squeeze(1)
            return self._boundary_dou(score, target_plane)

        prob = F.softmax(logits, dim=1)
        target_one_hot = self._one_hot(target, num_classes)

        loss = 0.0
        for c in range(num_classes):
            loss += self._boundary_dou(prob[:, c], target_one_hot[:, c])
        return loss / float(num_classes)
    
class SobelEdgeLoss(nn.Module):
    def __init__(self, from_logits=True, eps=1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def gradient_magnitude(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + self.eps)

    def normalize_per_sample(self, x):
        max_val = x.flatten(2).amax(dim=2).view(x.size(0), x.size(1), 1, 1)
        return x / max_val.clamp_min(self.eps)

    def forward(self, pred, target):
        target = target.float()

        if self.from_logits:
            pred = torch.sigmoid(pred)

        gt_edge = self.gradient_magnitude(target)
        pred_edge = self.gradient_magnitude(pred)

        gt_edge = self.normalize_per_sample(gt_edge)
        pred_edge = self.normalize_per_sample(pred_edge)

        # 只在 GT 边界附近更关注，避免全图背景主导
        weight = 1.0 + 2.0 * gt_edge.detach()

        loss = (torch.abs(pred_edge - gt_edge) * weight).mean()
        return loss

class FixedScheduledSegLossWrapper(nn.Module):
    def __init__(
        self,
        pos_weight: torch.Tensor,
        enable_dice: bool = True,
        enable_edge: bool = False,
        enable_area: bool = False,
        enable_active_contour: bool = False,
        ce_weight: float = 0.2,
        dice_weight: float = 0.8,
        edge_weight: float = 0.1,
        area_weight: float = 0.4,
        active_contour_weight: float = 0.4,
    ):
        super().__init__()
        self.enable_dice = enable_dice
        self.enable_edge = enable_edge
        self.enable_area = enable_area
        self.enable_active_contour = enable_active_contour

        self.w_ce = float(ce_weight)
        self.w_dice = float(dice_weight)
        self.w_edge = float(edge_weight)
        self.w_area = float(area_weight)
        self.w_active_contour = float(active_contour_weight)

        self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = MaskDiceLoss()
        self.edge = SobelEdgeLoss()
    @staticmethod
    def _flat_bt(x: torch.Tensor) -> torch.Tensor:
        # [B,T,1,H,W] -> [B*T,1,H,W]
        return x.view(-1, x.size(2), x.size(3), x.size(4)) if x.dim() == 5 else x

    def forward(self, net_output, target, current_epoch):

        total = 0.0
        log = {}

        target_seq = target
        logits_seq = net_output
        target = self._flat_bt(target)
        logits = self._flat_bt(net_output)

        ce_loss = self.ce(logits, target)
        total = total + ce_loss * self.w_ce
        log["ce"] = log.get("ce", 0.0) + ce_loss.detach()

        if self.enable_dice:
            dice_loss = self.dice(logits, target, sigmoid=True)
            total = total + dice_loss * self.w_dice
            log["dice"] = log.get("dice", 0.0) + dice_loss.detach()

        if self.enable_edge:
            edge_loss = self.edge(logits, target)
            total = total + edge_loss * self.w_edge
            log["edge"] = log.get("edge", 0.0) + edge_loss.detach()

        if self.enable_area and logits_seq.dim() == 5 and logits_seq.size(1) > 1:
            area_loss, area_logs = self.area(
                logits_seq,
                target_seq,
            )
            total = total + area_loss * self.w_area
            for key, value in area_logs.items():
                log[key] = log.get(key, 0.0) + value

        if self.enable_active_contour:
            active_contour_loss = self.active_contour(logits, target)
            total = total + active_contour_loss * self.w_active_contour
            log["active_contour"] = log.get("active_contour", 0.0) + active_contour_loss.detach()

        w = {
            "w_ce": self.w_ce,
            "w_dice": self.w_dice if self.enable_dice else 0.0,
            "w_edge": self.w_edge if self.enable_edge else 0.0,
            "w_area": self.w_area if self.enable_area else 0.0,
            "w_active_contour": self.w_active_contour if self.enable_active_contour else 0.0,
        }
        return total, log, w