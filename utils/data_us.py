import os
from random import randint
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torch.nn.functional as nnF
from typing import Callable
import os
import cv2
import pandas as pd
from numbers import Number
from typing import Container
from collections import defaultdict
# from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from torchvision.transforms import InterpolationMode

import random
from torchvision.utils import save_image

import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


def load_video_and_mask_file(
    img_path: str,
    anno_path: str,
    frame_length: int = 10,
    endpoint_supervision_only: bool = True,
):
    """
    采样策略：首尾必取，其余在 [1, T-2] 均匀取，frame_inds 基于源视频帧索引；phase 由源帧索引归一化，首尾必为 0/1。
    Returns:
        imgs: (F,3,H,W)
        select_masks: (F,...) masks
        ef, edv, esv, spacing
        phase: (F,) float32 in [0,1]
    """
    video = np.load(img_path, allow_pickle=True)  # (c, t, h, w)
    video = video.swapaxes(0, 1)  # (t, c, h, w)
    kpts_list = np.load(anno_path, allow_pickle=True)
    ef, edv, esv = kpts_list['ef'], kpts_list['edv'], kpts_list['esv']

    mask_list = kpts_list["fnum_mask"].tolist()
    if not mask_list:
        raise ValueError(f"Empty fnum_mask in annotation: {anno_path}")

    sorted_items = sorted(((int(k), v) for k, v in mask_list.items()), key=lambda x: x[0])
    idx_list = [k for k, _ in sorted_items]
    masks = [v for _, v in sorted_items]
    mask_by_idx = {k: v for k, v in sorted_items}

    total_frames = video.shape[0]
    requested_frame_length = int(frame_length) if frame_length is not None else 10

    ed_idx = int(np.clip(idx_list[0], 0, total_frames - 1))
    es_idx = int(np.clip(idx_list[-1], 0, total_frames - 1))
    if ed_idx > es_idx:
        ed_idx, es_idx = es_idx, ed_idx

    # 采样索引：严格在 [ED, ES] 区间内等间隔采样，保证时序单调。
    if requested_frame_length <= 0:
        frame_inds = list(range(ed_idx, es_idx + 1))
    elif requested_frame_length <= 2:
        frame_inds = [ed_idx, es_idx]
    else:
        frame_inds = np.linspace(ed_idx, es_idx, num=requested_frame_length, dtype=np.int32).tolist()
        frame_inds[0] = ed_idx
        frame_inds[-1] = es_idx

    # 收集帧
    frames = [video[min(int(i), total_frames - 1)] for i in frame_inds]
    select_masks = []
    ed_mask = mask_by_idx[idx_list[0]]
    es_mask = mask_by_idx[idx_list[-1]]
    for i in frame_inds:
        i_int = int(i)
        if endpoint_supervision_only:
            if i_int == ed_idx:
                select_masks.append(ed_mask)
            elif i_int == es_idx:
                select_masks.append(es_mask)
            else:
                select_masks.append(np.zeros_like(ed_mask))
        else:
            if i_int in mask_by_idx:
                select_masks.append(mask_by_idx[i_int])
            else:
                near_idx = min(idx_list, key=lambda x: abs(x - i_int))
                select_masks.append(mask_by_idx[near_idx])

    imgs = np.asarray(frames)
    select_masks = np.asarray(select_masks)

    # phase 统一按 ED->ES 区间归一化，首尾固定 0/1
    denom = max(1, es_idx - ed_idx)
    phase = (np.array(frame_inds, dtype=np.float32) - float(ed_idx)) / float(denom)
    phase[0] = 0.0
    phase[-1] = 1.0

    spacing = kpts_list['spacing']

    return imgs, select_masks, ef, edv, esv, spacing, phase

class CamusDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        joint_transform=None,
        img_size: int = 256,
        prompt: str = "click",
        class_id: int = 1,
        one_hot_mask: int = 0,
        frame_length: int = 2,
        disable_point_prompt: bool = False,
        point_numbers: int = 1,
        split_json: str = "camus_{split}_filenames.txt",
        video_dir: str = "videos",
        annotation_dir: str = "annotations",
        view: str = "all",
        endpoint_supervision_only: bool = True,
    ) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        self.frame_length = frame_length
        self.point_numbers = point_numbers
        self.prompt = prompt
        self.disable_point_prompt = disable_point_prompt
        self.img_size = img_size
        self.class_id = class_id
        self.video_root = os.path.join(dataset_path, video_dir, split)
        self.anno_root = os.path.join(dataset_path, annotation_dir, split)
        self.view = view
        self.endpoint_supervision_only = endpoint_supervision_only

        split_path = self._resolve_split_path(split_json, split)
        base_ids = self._read_split_list(split_path)
        self.ids = self._expand_view_ids(base_ids, view)
        self.ids = self._filter_existing_ids(self.ids)

        self.joint_transform = joint_transform

    def _resolve_split_path(self, split_json: str, split: str) -> str:
        default_name = f"camus_{split}_filenames.txt"
        candidate = split_json or default_name
        if "{split}" in candidate:
            candidate = candidate.format(split=split)
        candidate_path = candidate
        if not os.path.isabs(candidate_path):
            candidate_path = os.path.join(self.dataset_path, candidate_path)
        if os.path.isdir(candidate_path):
            candidate_path = os.path.join(candidate_path, default_name)
        if os.path.exists(candidate_path):
            return candidate_path
        return os.path.join(self.dataset_path, default_name)

    @staticmethod
    def _read_split_list(split_path: str) -> List[str]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")
        ids = []
        with open(split_path, "r") as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                ids.append(os.path.splitext(name)[0])
        return ids

    @staticmethod
    def _expand_view_ids(base_ids: List[str], view: str) -> List[str]:
        view_key = (view or "all").upper()
        expanded: List[str] = []
        for name in base_ids:
            upper = name.upper()
            if upper.endswith("_2CH") or upper.endswith("_4CH"):
                if view_key in {"ALL", upper[-3:]}:
                    expanded.append(name)
                continue
            if view_key in {"ALL", "2CH"}:
                expanded.append(f"{name}_2CH")
            if view_key in {"ALL", "4CH"}:
                expanded.append(f"{name}_4CH")
        return expanded

    def _filter_existing_ids(self, ids: List[str]) -> List[str]:
        valid = []
        for pid in ids:
            v = os.path.join(self.video_root, f"{pid}.npy")
            a = os.path.join(self.anno_root, f"{pid}.npz")
            if not os.path.exists(v) or not os.path.exists(a):
                continue
            valid.append(pid)
        if not valid:
            raise FileNotFoundError(
                f"No valid samples found. videos: {self.video_root}, annos: {self.anno_root}"
            )
        return valid

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def _to_torch_image(video_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(video_np)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.max() > 1.5:
                x = x / 255.0
        return x.clamp_(0.0, 1.0)

    @staticmethod
    def _to_torch_mask(mask_np: np.ndarray) -> torch.Tensor:
        m = torch.from_numpy(mask_np)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        m = (m > 0).float()
        return m

    def _resize_video_and_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = image.shape
        target = self.img_size
        if h == target and w == target:
            return image, mask
        image = torch.nn.functional.interpolate(
            image,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.nn.functional.interpolate(
            mask[:, None],
            size=(target, target),
            mode="nearest",
        )[:, 0]
        return image, mask
    def random_click_torch(self, mask: torch.Tensor, class_id: int = 1):
        """
        在指定的 mask 区域内随机选择一个点作为点击。
        """
        # 1. 确定目标区域 (eq)
        if mask.dtype != torch.bool:
            eq = (mask == class_id)
        else:
            eq = mask if class_id == 1 else ~mask

        # 2. 获取所有非零点的坐标 (N, 2)，每一行是 [y, x]
        idx = torch.nonzero(eq, as_tuple=False)
        
        point_label = 1
        
        # 3. 如果没有找到对应类别的像素点（空 mask 情况）
        if idx.numel() == 0:
            point_label = 0
            h, w = mask.shape[-2], mask.shape[-1]
            # 默认返回中心点，但 label 设为 0 (无效/背景)
            y = (h - 1) * 0.5
            x = (w - 1) * 0.5
            pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
            lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
            return pt, lab

        # 4. 随机选择一个索引
        # torch.randint 在 [0, idx.shape[0]) 范围内随机取一个数
        rand_i = torch.randint(0, idx.shape[0], (1,), device=mask.device)
        
        # 提取随机选中的点坐标
        # idx 存储的是 [row, col]，即 [y, x]
        selected_idx = idx[rand_i]  # 形状为 (1, 2)
        y = selected_idx[0, 0].float()
        x = selected_idx[0, 1].float()

        # 5. 构建返回格式 [[x, y]]
        pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
        lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
        
        return pt, lab
    def center_click_torch(self, mask: torch.Tensor, class_id: int = 1):
        if mask.dtype != torch.bool:
            eq = (mask == class_id)
        else:
            eq = mask if class_id == 1 else ~mask

        idx = torch.nonzero(eq, as_tuple=False)
        point_label = 1
        if idx.numel() == 0:
            point_label = 0
            h, w = mask.shape[-2], mask.shape[-1]
            y = (h - 1) * 0.5
            x = (w - 1) * 0.5
            pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
            lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
            return pt, lab

        y = idx[:, 0].float().mean()
        x = idx[:, 1].float().mean()
        pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
        lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
        return pt, lab

    def __getitem__(self, i: int):
        patient_id = self.ids[i]
        video_path = os.path.join(self.video_root, f"{patient_id}.npy")
        anno_path = os.path.join(self.anno_root, f"{patient_id}.npz")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        imgs, masks, _, _, _, spacing, phase = load_video_and_mask_file(
            video_path,
            anno_path,
            frame_length=self.frame_length,
            endpoint_supervision_only=self.endpoint_supervision_only,
        )
        masks[masks == 1] = 1
        image = self._to_torch_image(imgs)
        m = torch.from_numpy(masks)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        mask = (m == 1).float()
        # print(mask[0])
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)
        image, mask = self._resize_video_and_mask(image, mask)

        if self.disable_point_prompt:
            f_len = mask.shape[0]
            pt = torch.empty((f_len, 0, 2), dtype=torch.float32)
            point_labels = torch.empty((f_len, 0), dtype=torch.int64)
        else:
            pts = []
            pls = []
            for f in range(mask.shape[0]):
                p, l = self.random_click_torch(mask[f], class_id=1)

                if self.point_numbers > 1:
                    p0, l0 = self.random_click_torch(mask[f], class_id=1)
                    p = p0.repeat(self.point_numbers, 1)
                    l = l0.repeat(self.point_numbers)

                pts.append(p)
                pls.append(l)

            pt = torch.stack(pts, dim=0)
            point_labels = torch.stack(pls, dim=0)

        if self.one_hot_mask and self.one_hot_mask > 0:
            m_long = mask.long()
            mask = torch.zeros((mask.shape[0], self.one_hot_mask, mask.shape[1], mask.shape[2]), dtype=torch.float32)
            mask.scatter_(1, m_long[:, None, :, :], 1.0)

        phase = torch.from_numpy(np.asarray(phase, dtype=np.float32)).float()

        return {
            "image": image,
            "label": mask,
            "point_labels":point_labels,
            "pt": pt ,
            "image_name": patient_id,
            "class_id": self.class_id,
            "phase": phase,
            "spacing": spacing,
        }

class EchoVideoDataset(Dataset):

    def __init__(self,
                 dataset_path: str,
                 split='train',
                 joint_transform: Callable = None,
                 img_size=112,
                 prompt="click",
                 class_id=1,
                 one_hot_mask: int = False,
                 frame_length: int = 2,
                 disable_point_prompt: bool = True,
                 point_numbers: int = 1,
                 endpoint_supervision_only: bool = True) -> None:
        self.dataset_path = dataset_path
        self.one_hot_mask = one_hot_mask
        self.split = split
        self.frame_length = frame_length
        self.point_numbers = point_numbers
        self.endpoint_supervision_only = endpoint_supervision_only
        self.ids = []
        for _, _, files in os.walk(os.path.join(dataset_path, 'videos',
                                                split)):
            self.ids = files

        # id_list_file = os.path.join(dataset_path, '{0}.txt'.format(split))
        # self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.prompt = prompt
        self.disable_point_prompt = disable_point_prompt
        self.img_size = img_size
        self.class_id = class_id
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.ids)
    def center_click_torch(self, mask: torch.Tensor, class_id: int = 1):
            if mask.dtype != torch.bool:
                eq = (mask == class_id)
            else:
                eq = mask if class_id == 1 else ~mask

            idx = torch.nonzero(eq, as_tuple=False)
            point_label = 1
            if idx.numel() == 0:
                point_label = 0
                h, w = mask.shape[-2], mask.shape[-1]
                y = (h - 1) * 0.5
                x = (w - 1) * 0.5
                pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
                lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
                return pt, lab

            y = idx[:, 0].float().mean()
            x = idx[:, 1].float().mean()
            pt = torch.tensor([[x, y]], device=mask.device, dtype=torch.float32)
            lab = torch.tensor([point_label], device=mask.device, dtype=torch.int64)
            return pt, lab
    @staticmethod
    def _to_torch_image(video_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(video_np)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        else:
            x = x.float()
            if x.max() > 1.5:
                x = x / 255.0
        return x.clamp_(0.0, 1.0)
    def _resize_video_and_mask(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = image.shape
        target = self.img_size
        if h == target and w == target:
            return image, mask
        image = torch.nn.functional.interpolate(
            image,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )
        mask = torch.nn.functional.interpolate(
            mask[:, None],
            size=(target, target),
            mode="nearest",
        )[:, 0]
        return image, mask

    def __getitem__(self, i):
        filename = self.ids[i]
        prefix, _ = os.path.splitext(filename)
        class_id = 1

        img_path = os.path.join(os.path.join(self.dataset_path, 'videos'), self.split)
        label_path = os.path.join(os.path.join(self.dataset_path, 'annotations'), self.split)
        
        # 此时读取出来的是 numpy array
        imgs, masks, _, _, _, spacing, phase  = load_video_and_mask_file(
            img_path=os.path.join(img_path, prefix + '.npy'),
            anno_path=os.path.join(label_path, prefix + '.npz'),
            frame_length=self.frame_length,
            endpoint_supervision_only=self.endpoint_supervision_only,
        )
        masks[masks == 1] = 1
        image = self._to_torch_image(imgs)
        m = torch.from_numpy(masks)
        if m.ndim == 4 and m.shape[1] == 1:
            m = m[:, 0]
        mask = (m == 1).float()
        # print(mask[0])
        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)
        image, mask = self._resize_video_and_mask(image, mask)

        # --------- make the point prompt -----------------
        if self.disable_point_prompt:
            f_len = mask.shape[0]
            pt = torch.empty((f_len, 0, 2), dtype=torch.float32)
            point_labels = torch.empty((f_len, 0), dtype=torch.int64)
        else:
            pts = []
            pls = []
            for f in range(mask.shape[0]):
                p, l = self.center_click_torch(mask[f], class_id=1)

                if self.point_numbers > 1:
                    p0, l0 = self.center_click_torch(mask[f], class_id=1)
                    p = p0.repeat(self.point_numbers, 1)
                    l = l0.repeat(self.point_numbers)

                pts.append(p)
                pls.append(l)

            pt = torch.stack(pts, dim=0)
            point_labels = torch.stack(pls, dim=0)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            # 注意：此处用的是 long()，前面已经将其转为 Tensor，因此不会报错了
            mask = torch.zeros((self.one_hot_mask, mask.shape[1],
                                mask.shape[2])).scatter_(0, mask.long(), 1)

        return {
            "image": image,
            "label": mask,
            "point_labels": point_labels,
            "pt": pt,
            "image_name": filename,
            "class_id": class_id,
            "phase": torch.from_numpy(np.asarray(phase, dtype=np.float32)).float(),
            "spacing": spacing,
        }
