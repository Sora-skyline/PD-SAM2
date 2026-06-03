import os
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
# import torch
from torch.autograd import Variable
from tqdm import tqdm

from medpy.metric.binary import hd95 as medpy_hd95
from medpy.metric.binary import assd as medpy_assd
import utils.metrics as metrics
from utils.visualization import  visual_segmentation_npy

import time
import numpy as np
import torch
import torch.distributed as dist
# EF tool
def compute_ef_from_masks(mask_seq):
    """
    mask_seq: [B, T, H, W] or [B, T, 1, H, W]
    return: ef [B]
    用前景像素数近似面积/容积
    """
    if mask_seq.dim() == 5:
        mask_seq = mask_seq.squeeze(2)  # [B,T,H,W]

    B, T, H, W = mask_seq.shape
    flat = mask_seq.float().view(B, T, -1).sum(dim=-1)  # [B,T]

    # 不假设首尾一定是 ED/ES，直接取最大/最小
    ed = flat.max(dim=1).values
    es = flat.min(dim=1).values

    ef = (ed - es) / (ed + 1e-8) * 100.0
    return ef


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    A = ((x - x.mean()) * (y - y.mean())).mean()
    B = x.std() * y.std()
    return float(A / (B + 1e-8))


def bias_and_std(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0, 0.0
    diff = pred - gt
    bias = diff.mean()
    std = diff.std()
    return float(bias), float(std)


def mae(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0
    return float(np.abs(pred - gt).mean())


def rmse(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0
    return float(np.sqrt(((pred - gt) ** 2).mean()))

def ensure_video_like(datapack, device, mode: str = "video", target_len: int = None):
    """
    Normalize batch to video-like tensors.
    Returns:
        images: [B, T, 3, H, W]
        masks:  [B, T, 1, H, W]
        pts:    always None (prompt-free training)
        phases: [B, T] or None
    """
    imgs = datapack["image"].to(dtype=torch.float32, device=device)
    masks = datapack["label"].to(dtype=torch.float32, device=device)
    phases = datapack.get("phase", None).to(dtype=torch.float32, device=device)
    point_labels = datapack.get("point_labels", None).to(dtype=torch.float32, device=device)
    pts = datapack["pt"].to(dtype=torch.float32, device=device)
    pts = (pts, point_labels)
    if imgs.dim() == 4:  # [B, C, H, W] -> [B,1,C,H,W]
        imgs = imgs.unsqueeze(1)
    if masks.dim() == 3:  # [B,H,W] -> [B,1,H,W]
        masks = masks.unsqueeze(1)
    if masks.dim() == 4:  # [B,T,H,W] -> [B,T,1,H,W]
        masks = masks.unsqueeze(2)

    if phases is not None:
        phases = phases.to(dtype=torch.float32, device=device)
        if phases.dim() == 1:  # [T] -> [1,T]
            phases = phases.unsqueeze(0)
        if phases.shape[0] != imgs.shape[0]:
            phases = phases.expand(imgs.shape[0], -1)
        if phases.shape[1] != imgs.shape[1]:
            if phases.shape[1] == 1:
                phases = phases.repeat(1, imgs.shape[1])
            else:
                steps = torch.linspace(0, 1, steps=imgs.shape[1], device=device)
                phases = steps.unsqueeze(0).expand(imgs.shape[0], -1)
    else:
        phases = torch.linspace(0, 1, steps=imgs.shape[1], device=device).unsqueeze(0).expand(imgs.shape[0], -1) if mode == "image" else None

    return imgs, masks, pts, phases

def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def _reduce_metrics_weighted(local_results: dict, local_count: int) -> dict:
    """
    Weighted reduce across ranks:
      global = sum(value * count) / sum(count)
    local_count: number of evaluated samples on this rank (e.g., buffer length)
    """
    if not _is_dist_avail_and_initialized():
        return local_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # total count
    cnt = torch.tensor([float(local_count)], device=device)
    dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
    total_count = float(cnt.item())
    if total_count <= 0:
        total_count = 1.0

    reduced = {}
    for k, v in local_results.items():
        t = torch.tensor([float(v) * float(local_count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = float(t.item() / total_count)

    return reduced

import os
import time
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)

import os
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
# import torch
from torch.autograd import Variable
from tqdm import tqdm

from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
from utils.visualization import  visual_segmentation_npy

import time
import numpy as np
import torch
import torch.distributed as dist

from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, SurfaceDistanceMetric
# EF tool
def compute_ef_from_masks(mask_seq):
    """
    mask_seq: [B, T, H, W] or [B, T, 1, H, W]
    return: ef [B]
    用前景像素数近似面积/容积
    """
    if mask_seq.dim() == 5:
        mask_seq = mask_seq.squeeze(2)  # [B,T,H,W]

    B, T, H, W = mask_seq.shape
    flat = mask_seq.float().view(B, T, -1).sum(dim=-1)  # [B,T]

    # 不假设首尾一定是 ED/ES，直接取最大/最小
    ed = flat.max(dim=1).values
    es = flat.min(dim=1).values

    ef = (ed - es) / (ed + 1e-8) * 100.0
    return ef


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    A = ((x - x.mean()) * (y - y.mean())).mean()
    B = x.std() * y.std()
    return float(A / (B + 1e-8))


def bias_and_std(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0, 0.0
    diff = pred - gt
    bias = diff.mean()
    std = diff.std()
    return float(bias), float(std)


def mae(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0
    return float(np.abs(pred - gt).mean())


def rmse(gt, pred):
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(gt) == 0:
        return 0.0
    return float(np.sqrt(((pred - gt) ** 2).mean()))

def ensure_video_like(datapack, device, mode: str = "video", target_len: int = None):
    """
    Normalize batch to video-like tensors.
    Returns:
        images: [B, T, 3, H, W]
        masks:  [B, T, 1, H, W]
        pts:    always None (prompt-free training)
        phases: [B, T] or None
    """
    imgs = datapack["image"].to(dtype=torch.float32, device=device)
    masks = datapack["label"].to(dtype=torch.float32, device=device)
    phases = datapack.get("phase", None).to(dtype=torch.float32, device=device)
    point_labels = datapack.get("point_labels", None).to(dtype=torch.float32, device=device)
    pts = datapack["pt"].to(dtype=torch.float32, device=device)
    pts = (pts, point_labels)
    if imgs.dim() == 4:  # [B, C, H, W] -> [B,1,C,H,W]
        imgs = imgs.unsqueeze(1)
    if masks.dim() == 3:  # [B,H,W] -> [B,1,H,W]
        masks = masks.unsqueeze(1)
    if masks.dim() == 4:  # [B,T,H,W] -> [B,T,1,H,W]
        masks = masks.unsqueeze(2)

    T_curr = imgs.shape[1]
    if target_len is None:
        target_len = T_curr
    if T_curr != target_len:
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, target_len, 1, 1, 1)
        if masks.shape[1] == 1:
            masks = masks.repeat(1, target_len, 1, 1, 1)

    if phases is not None:
        phases = phases.to(dtype=torch.float32, device=device)
        if phases.dim() == 1:  # [T] -> [1,T]
            phases = phases.unsqueeze(0)
        if phases.shape[0] != imgs.shape[0]:
            phases = phases.expand(imgs.shape[0], -1)
        if phases.shape[1] != imgs.shape[1]:
            if phases.shape[1] == 1:
                phases = phases.repeat(1, imgs.shape[1])
            else:
                steps = torch.linspace(0, 1, steps=imgs.shape[1], device=device)
                phases = steps.unsqueeze(0).expand(imgs.shape[0], -1)
    else:
        phases = torch.linspace(0, 1, steps=imgs.shape[1], device=device).unsqueeze(0).expand(imgs.shape[0], -1) if mode == "image" else None
    return imgs, masks, pts, phases

def _is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def _reduce_metrics_weighted(local_results: dict, local_count: int) -> dict:
    """
    Weighted reduce across ranks:
      global = sum(value * count) / sum(count)
    local_count: number of evaluated samples on this rank (e.g., buffer length)
    """
    if not _is_dist_avail_and_initialized():
        return local_results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # total count
    cnt = torch.tensor([float(local_count)], device=device)
    dist.all_reduce(cnt, op=dist.ReduceOp.SUM)
    total_count = float(cnt.item())
    if total_count <= 0:
        total_count = 1.0

    reduced = {}
    for k, v in local_results.items():
        t = torch.tensor([float(v) * float(local_count)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[k] = float(t.item() / total_count)

    return reduced


def _select_metric_frame_indices(num_frames: int, opt, args) -> list:
    if num_frames <= 0:
        return []

    metric_frame_mode = getattr(args, "metric_frame_mode", None)
    if metric_frame_mode is None:
        metric_frame_mode = "endpoints" if getattr(opt, "semi", True) else "clip"
    metric_frame_mode = str(metric_frame_mode).lower()

    if metric_frame_mode == "endpoints":
        return sorted(set([0, num_frames - 1]))
    if metric_frame_mode == "all":
        return list(range(num_frames))
    if metric_frame_mode != "clip":
        raise ValueError(
            f"Unsupported metric_frame_mode={metric_frame_mode!r}; "
            "expected 'endpoints', 'clip', or 'all'."
        )

    metric_len = getattr(args, "metric_frame_length", None)
    if metric_len is None:
        frame_length = getattr(args, "frame_length", None)
        metric_len = frame_length if frame_length and frame_length > 0 else 10
    metric_len = int(metric_len)

    if metric_len <= 0 or metric_len >= num_frames:
        return list(range(num_frames))
    if metric_len <= 2:
        return sorted(set([0, num_frames - 1]))

    frame_indices = np.linspace(0, num_frames - 1, num=metric_len, dtype=np.int32).tolist()
    frame_indices[0] = 0
    frame_indices[-1] = num_frames - 1
    return sorted(set(int(i) for i in frame_indices))

def _get_memsam_spacing(datapack: dict, sample_idx: int):
    spacing = datapack.get("spacing", None)
    if spacing is None:
        return None
    if torch.is_tensor(spacing):
        spacing = spacing.detach().cpu().numpy()
    spacing = np.asarray(spacing)
    if spacing.ndim == 0:
        return float(spacing)
    sample_spacing = spacing[sample_idx] if spacing.ndim > 1 else spacing
    sample_spacing = np.asarray(sample_spacing, dtype=np.float64).reshape(-1)
    if sample_spacing.size >= 2:
        return sample_spacing[:2][::-1]
    if sample_spacing.size == 1:
        return float(sample_spacing[0])
    return None

def _memsam_frame_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, spacing):
    h, w = pred_mask.shape
    pred_i = np.zeros((1, h, w), dtype=np.uint8)
    gt_i = np.zeros((1, h, w), dtype=np.uint8)
    pred_i[pred_mask[None, :, :] == 1] = 255
    gt_i[gt_mask[None, :, :] == 1] = 255

    tp, fp, _, fn = metrics.get_matrix(pred_i, gt_i)
    smooth = 1e-5
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (fp + tp + fn + smooth)

    pred_has_obj = bool(pred_i.any())
    gt_has_obj = bool(gt_i.any())
    if pred_has_obj and gt_has_obj:
        hd95 = medpy_hd95(pred_i[0], gt_i[0], voxelspacing=spacing)
        assd = medpy_assd(pred_i[0], gt_i[0], voxelspacing=spacing)
    elif not pred_has_obj and not gt_has_obj:
        hd95 = 0.0
        assd = 0.0
    else:
        if spacing is None:
            y_spacing, x_spacing = 1.0, 1.0
        else:
            spacing_arr = np.asarray(spacing, dtype=np.float64).reshape(-1)
            y_spacing = float(spacing_arr[0]) if spacing_arr.size >= 1 else 1.0
            x_spacing = float(spacing_arr[1]) if spacing_arr.size >= 2 else y_spacing
        hd95 = float(np.hypot((h - 1) * y_spacing, (w - 1) * x_spacing))
        assd = hd95

    return (
        float(np.mean(dice)),
        float(np.mean(iou)),
        float(hd95),
        float(assd),
    )


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8)


def _apply_largest_connected_component(pred_np: np.ndarray) -> np.ndarray:
    if pred_np.size == 0:
        return pred_np
    out = np.empty_like(pred_np, dtype=np.uint8)
    flat_in = pred_np.reshape(-1, pred_np.shape[-2], pred_np.shape[-1])
    flat_out = out.reshape(-1, out.shape[-2], out.shape[-1])
    for idx, mask in enumerate(flat_in):
        flat_out[idx] = _largest_connected_component(mask)
    return out


def eval_camus_test(valloader, model, criterion, opt, args, epoch=None, save_vis=False):
    """
    Baseline-style global eval:
      - default semi=True computes metrics/loss on first & last frames
      - default semi=False computes metrics/loss on the configured clip frames
      - args.metric_frame_mode can override the default frame set
      - visual output is written for every loaded frame
      - MemSAM-style frame-wise metric aggregation
      - return dict like baseline global_metrics
    """
    gt_ef_list = []
    pred_ef_list = []

    save_vis = bool(save_vis or getattr(args, "save_vis", False))
    # barrier like baseline
    if _is_dist_avail_and_initialized():
        dist.barrier()

    prev_mode = model.training
    model.eval()

    dice_values = []
    iou_values = []
    hd95_values = []
    assd_values = []

    val_losses = 0.0
    val_raw_losses = {}
    num_batches = 0
    sum_time = 0.0
    total_frames = 0
    metric_frame_count = 0
    output_frame_count = 0
    eval_threshold = float(getattr(args, "eval_threshold", 0.6))
    eval_largest_component = bool(getattr(args, "eval_largest_component", False))

    per_image_metrics = []
    low_dice_cases = []
    is_main = (not _is_dist_avail_and_initialized()) or dist.get_rank() == 0
    loader_iter = tqdm(valloader, desc="Eval", leave=False) if is_main else valloader

    with torch.no_grad():
        for batch_idx, datapack in enumerate(loader_iter):
            imgs, masks,pts, phases = ensure_video_like(
                datapack,
                device=opt.device,
                mode=args.mode,
                target_len=None if args.frame_length == -1 else args.frame_length,
            )

            B, T, C, H, W = imgs.shape
            total_frames += int(B * T)
            image_names = datapack.get("image_name", None)
            if isinstance(image_names, str):
                image_names = [image_names] * B
            elif image_names is not None:
                image_names = list(image_names)
            if image_names is not None and len(image_names) != B:
                image_names = [f"sample_{batch_idx}_{i}" for i in range(B)]

            model_kwargs = {}
            if bool(getattr(args, "save_memsam_featmap_vis", False)):
                model_kwargs["vis_patient_names"] = image_names

            start = time.time()
            pred = model(imgs, phases, pts, **model_kwargs)
            sum_time += (time.time() - start)

            metric_frame_indices = _select_metric_frame_indices(T, opt, args)
            save_frame_indices = list(range(T))
            metric_frame_count = len(metric_frame_indices)
            output_frame_count = len(save_frame_indices)

            pred_for_loss = pred
            masks_for_loss = masks
            if metric_frame_indices:
                pred_for_loss = pred[:, metric_frame_indices, ...]
                masks_for_loss = masks[:, metric_frame_indices, ...]

            val_loss, raw_losses, _ = criterion(
                pred_for_loss, masks_for_loss, current_epoch=epoch
            )
            val_losses += float(val_loss.item())
            num_batches += 1
            for k, v in raw_losses.items():
                if k not in val_raw_losses:
                    val_raw_losses[k] = 0.0
                val_raw_losses[k] += float(v.item())

            # (B,T,H,W)
            pred = pred.view(B, T, H, W)
            masks = masks.view(B, T, H, W)
            
            pred_seq_bin = (torch.sigmoid(pred) > eval_threshold).float()
            gt_seq_bin = (masks > 0.5).float()

            pred_np = pred_seq_bin.detach().cpu().numpy().astype(np.uint8)
            if eval_largest_component:
                pred_np = _apply_largest_connected_component(pred_np)
            gt_np = gt_seq_bin.detach().cpu().numpy().astype(np.uint8)

            pred_metric_tensor = torch.from_numpy(pred_np[:, metric_frame_indices, ...])
            gt_metric_tensor = torch.from_numpy(gt_np[:, metric_frame_indices, ...])
            pred_ef_batch = compute_ef_from_masks(pred_metric_tensor)
            gt_ef_batch = compute_ef_from_masks(gt_metric_tensor)

            pred_ef_list.extend(pred_ef_batch.detach().cpu().numpy().tolist())
            gt_ef_list.extend(gt_ef_batch.detach().cpu().numpy().tolist())

            metric_names = image_names or [f"sample_{batch_idx}_{i}" for i in range(B)]

            for j_idx in range(B):
                spacing = _get_memsam_spacing(datapack, j_idx)
                img_name = metric_names[j_idx]
                for frame_i in metric_frame_indices:
                    dice_val, iou_val, hd95_val, assd_val = _memsam_frame_metrics(
                        pred_np[j_idx, frame_i],
                        gt_np[j_idx, frame_i],
                        spacing,
                    )
                    dice_values.append(dice_val)
                    iou_values.append(iou_val)
                    hd95_values.append(hd95_val)
                    assd_values.append(assd_val)

                    if save_vis and is_main:
                        frame_metrics = {
                            "image_name": img_name,
                            "frame_idx": int(frame_i),
                            "dice": dice_val,
                            "iou": iou_val,
                            "hd95": hd95_val,
                            "assd": assd_val,
                        }
                        per_image_metrics.append(frame_metrics)
                        if dice_val < 0.9:
                            low_dice_cases.append(frame_metrics)

            if save_vis and is_main:
                logits_np = pred.detach().cpu().numpy()
                for frame_i in save_frame_indices:
                    for j_idx in range(B):
                        img_name = metric_names[j_idx]
                        visual_segmentation_npy(
                            pred_np[j_idx, frame_i],
                            gt_np[j_idx, frame_i],
                            img_name,
                            opt,
                            imgs[j_idx:j_idx + 1, frame_i, :, :, :],
                            frameidx=frame_i,
                            patient_name=img_name,
                            mask_logits=logits_np[j_idx, frame_i],
                        )

    local_counts = len(dice_values)

    def _mean_or_zero(values):
        if not values:
            return 0.0
        value = float(np.mean(np.asarray(values, dtype=np.float64)))
        return value if np.isfinite(value) else 0.0

    local_results = {
        "dice": _mean_or_zero(dice_values),
        "iou": _mean_or_zero(iou_values),
        "hd95": _mean_or_zero(hd95_values),
        "assd": _mean_or_zero(assd_values),
    }
    global_metrics = _reduce_metrics_weighted(local_results, local_counts)

    # optional: add loss/fps if你想 log（baseline里通常也会记录）
    avg_loss = val_losses / max(num_batches, 1)
    avg_fps = 0.0
    if total_frames > 0 and sum_time > 0:
        avg_fps = float(total_frames) / float(sum_time)

    if save_vis and per_image_metrics and ((not _is_dist_avail_and_initialized()) or dist.get_rank() == 0):
        out_dir = os.path.join(opt.result_path, "vis", opt.modelname)
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, f"{args.task}_image_metrics.json")
        low_dice_path = os.path.join(out_dir, f"{args.task}_low_dice.json")
        with open(metrics_path, "w") as f:
            json.dump(per_image_metrics, f, indent=4)
        with open(low_dice_path, "w") as f:
            json.dump(low_dice_cases, f, indent=4)
# EF
    if _is_dist_avail_and_initialized():
        world_size = dist.get_world_size()
        gathered_gt = [None for _ in range(world_size)]
        gathered_pred = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gt, gt_ef_list)
        dist.all_gather_object(gathered_pred, pred_ef_list)

        gt_ef_array = np.array([x for sub in gathered_gt for x in sub], dtype=np.float64)
        pred_ef_array = np.array([x for sub in gathered_pred for x in sub], dtype=np.float64)
    else:
        gt_ef_array = np.array(gt_ef_list, dtype=np.float64)
        pred_ef_array = np.array(pred_ef_list, dtype=np.float64)

    if len(gt_ef_array) > 0:
        ef_bias, ef_std = bias_and_std(gt_ef_array, pred_ef_array)
        ef_corr = corr(gt_ef_array, pred_ef_array)
        ef_mae = mae(gt_ef_array, pred_ef_array)
        ef_rmse = rmse(gt_ef_array, pred_ef_array)
    else:
        ef_bias, ef_std, ef_corr, ef_mae, ef_rmse = 0.0, 0.0, 0.0, 0.0, 0.0

    global_metrics.update(
        {
            "loss": avg_loss,
            "fps": avg_fps,
            "count": local_counts,
            "raw_losses": val_raw_losses,
            "steps": num_batches,
            "ef_bias": ef_bias,
            "ef_std": ef_std,
            "ef_corr": ef_corr,
            "ef_mae": ef_mae,
            "ef_rmse": ef_rmse,
            "frame_eval_mode": getattr(
                args,
                "metric_frame_mode",
                "endpoints" if getattr(opt, "semi", True) else "clip",
            ),
            "frame_output_mode": "full_sequence" if getattr(args, "infer_all_frames", False) else "loaded_clip",
            "metric_frame_count": metric_frame_count,
            "output_frame_count": output_frame_count,
            "eval_threshold": eval_threshold,
            "eval_largest_component": eval_largest_component,
        }
    )

    if _is_dist_avail_and_initialized():
        dist.barrier()

    model.train(prev_mode)
    return global_metrics
