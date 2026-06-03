import cv2
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import Tensor


def find_contours(mask: Tensor):
    h,w = mask.shape
    if isinstance(mask, Tensor): 
        mask = mask.numpy().astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    assert len(contours) == 1
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return edge


def find_contour_points(mask: Tensor):
    '''
        mask: (h,w), 0 or 1
        return: contours (n,2)
                the x,y of the points 
    '''
    h,w = mask.shape
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy().astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    edge = np.zeros((h,w), dtype=np.uint8)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    # assert len(contours) == 1
    if len(contours) != 1:
        return np.array([])
    
    contours = contours[0].squeeze()
    edge[contours[:,1], contours[:,0]] = 1

    # check
    assert edge.sum() == (edge*mask).sum()
    return contours


def hausdorff_distance(mask1: Tensor, mask2: Tensor, percentile: int = 95):
    if isinstance(mask1, torch.Tensor) and mask1.device == 'cuda':
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()

    contours1 = find_contour_points(mask1)
    contours2 = find_contour_points(mask2)
    if contours1.size == 0 or contours2.size == 0:
        return 0

    dist = cdist(contours1, contours2)
    dist = np.concatenate((np.min(dist, axis=0), np.min(dist, axis=1)))
    assert percentile >= 0 and percentile <= 100, 'percentile invaild'
    hausdorff_dist = np.percentile(dist, percentile)

    return hausdorff_dist


def corr(x, y):
    '''
        x : gt
        y : pred
        A = mean( (y_real - mean(y_real)) * (y_predict - mean(y_predict)) )
        B = std(y_real) * std(y_predict)
        corr = A / B
    '''
    A = ((x - x.mean()) * (y - y.mean())).mean()
    B = std(x) * std(y)
    corr = A / B
    return corr

def bias(x, y):
    '''
        x : gt
        y : pred
        bias = sum( abs(y_real - y_predict) ) / len( y_real )
    '''
    return (abs(x - y)).mean()

def std(x):
    '''
        A = (y - mean(y)) * (y - mean(y))
        std = sqrt( sum(A) / n )
    '''
    # A = ((x - x.mean()) * (x - x.mean())).mean()
    # std = np.sqrt(A)
    return np.std(x)

def mae(gt, pred):
    # 绝对误差的平均值（你现在的 bias）
    diff_abs = np.abs(pred - gt)
    return diff_abs.mean()

def bias_and_std(gt, pred):
    # 论文表里的 bias ± std
    diff = pred - gt               # signed error
    bias = diff.mean()
    std  = diff.std()
    return bias, std

def draw_sem_seg_by_cv2_sum(image, gt_sem_seg, pred_sem_seg, palette, alpha=0.8):
    """
    image: [3,H,W] float(0~1) or uint8(0~255)
    gt/pred: [H,W] 0/1 or 0/255
    alpha: color-mask weight. MemSAM uses image=0.2, mask=0.8.
    """
    # --- 1) 统一 image 到 uint8 0~255 ---
    if image.dtype != np.uint8:
        img = image.astype(np.float32)
        # 如果是 0~1，拉到 0~255
        if img.size > 0 and img.max() <= 1.0 + 1e-6:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img = image.copy()

    gt_sem_seg = (gt_sem_seg > 0).astype(np.uint8)
    pred_sem_seg = (pred_sem_seg > 0).astype(np.uint8)

    mask = (2 * pred_sem_seg + gt_sem_seg).squeeze()  # 0..3

    # --- 2) color_mask 用 uint8 ---
    color_mask = np.zeros_like(img, dtype=np.uint8)
    ids = np.unique(mask)
    for idx in ids:
        color_mask[0][mask == idx] = palette[idx][0]
        color_mask[1][mask == idx] = palette[idx][1]
        color_mask[2][mask == idx] = palette[idx][2]

    # --- 3) 融合：MemSAM 论文图风格，颜色覆盖更重 ---
    results = cv2.addWeighted(img, 1 - alpha, color_mask, alpha, 0)

    # --- 4) 只在 mask != 0 的区域覆盖融合结果 ---
    mask3 = np.expand_dims(mask, 0).repeat(3, axis=0)
    out = img.copy()
    out[mask3 != 0] = results[mask3 != 0]

    return out
