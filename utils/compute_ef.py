'''
    from script_camus_ef.ipynb
    download from https://www.creatis.insa-lyon.fr/Challenge/camus/evaluationSegmentation.html
'''

import logging
from pathlib import Path
from typing import Any, Dict, Tuple
from skimage.measure import regionprops

import numpy as np
import PIL
import SimpleITK as sitk
from PIL.Image import Resampling
from skimage.measure import find_contours
import os
import matplotlib.pyplot as plt
from PIL import Image
import os
logger = logging.getLogger(__name__)

def sitk_load(filepath: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Loads an image using SimpleITK and returns the image and its metadata.

    Args:
        filepath: Path to the image.

    Returns:
        - ([N], H, W), Image array.
        - Collection of metadata.
    """
    # Load image and save info
    image = sitk.ReadImage(str(filepath))
    info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

    # Extract numpy array from the SimpleITK image object
    im_array = np.squeeze(sitk.GetArrayFromImage(image))

    return im_array, info

def resize_image(image: np.ndarray, size: Tuple[int, int], resample: Resampling = Resampling.NEAREST) -> np.ndarray:
    """Resizes the image to the specified dimensions.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        size: Width (W') and height (H') dimensions of the resized image to output.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized to the specified dimensions.
    """
    resized_image = np.array(PIL.Image.fromarray(image).resize(size, resample=resample))
    return resized_image

def resize_image_to_isotropic(
    image: np.ndarray, spacing: Tuple[float, float], resample: Resampling = Resampling.NEAREST
) -> np.ndarray:
    """Resizes the image to attain isotropic spacing, by resampling the dimension with the biggest voxel size.

    Args:
        image: (H, W), Input image to resize. Must be in a format supported by PIL.
        spacing: Size of the image's pixels along each (height, width) dimension.
        resample: Resampling filter to use.

    Returns:
        (H', W'), Input image resized so that the spacing is isotropic, and the isotropic value of the new spacing.
    """
    scaling = np.array(spacing) / min(spacing)
    new_height, new_width = (np.array(image.shape) * scaling).round().astype(int)
    return resize_image(image, (new_width, new_height), resample=resample), min(spacing)
def _save_lv_debug_sequence(
    imgs,
    patient_name: str,
    save_dir: str = "debug",
):
    """把 4 张 numpy 图像横向拼接成一张并保存"""
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return

    pil_imgs = [Image.fromarray(im) for im in imgs]
    w, h = pil_imgs[0].size

    canvas = Image.new("RGB", (w * len(pil_imgs), h), (0, 0, 0))
    for i, im in enumerate(pil_imgs):
        canvas.paste(im, (i * w, 0))

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{patient_name}_2ch4ch_ed_es.png")
    canvas.save(out_path)
    print(f"saved LV debug sequence to {out_path}")
def compute_left_ventricle_volumes(
    a2c_ed: np.ndarray,
    a2c_es: np.ndarray,
    a2c_voxelspacing: Tuple[float, float],
    a4c_ed: np.ndarray,
    a4c_es: np.ndarray,
    a4c_voxelspacing: Tuple[float, float],
    patient_name
) -> Tuple[float, float]:

    a2c_ed_diameters, a2c_ed_step_size, img_a2c_ed = _compute_diameters_debug(
        a2c_ed, a2c_voxelspacing
    )
    a2c_es_diameters, a2c_es_step_size, img_a2c_es = _compute_diameters_debug(
        a2c_es, a2c_voxelspacing
    )
    a4c_ed_diameters, a4c_ed_step_size, img_a4c_ed = _compute_diameters_debug(
        a4c_ed, a4c_voxelspacing
    )
    a4c_es_diameters, a4c_es_step_size, img_a4c_es = _compute_diameters_debug(
        a4c_es, a4c_voxelspacing
    )

    step_size = max((a2c_ed_step_size, a2c_es_step_size, a4c_ed_step_size, a4c_es_step_size))

    ed_volume = _compute_left_ventricle_volume_by_instant(a2c_ed_diameters, a4c_ed_diameters, step_size)
    es_volume = _compute_left_ventricle_volume_by_instant(a2c_es_diameters, a4c_es_diameters, step_size)

    # 只保留这四个视图的 debug 图，拼成一个序列保存
    _save_lv_debug_sequence(
        [img_a2c_ed, img_a2c_es, img_a4c_ed, img_a4c_es],
        patient_name=patient_name,
        save_dir="debug"
    )

    return ed_volume, es_volume

def _compute_left_ventricle_volume_by_instant(
    a2c_diameters: np.ndarray, a4c_diameters: np.ndarray, step_size: float
) -> float:
    """Compute left ventricle volume using Biplane Simpson's method.

    Args:
        a2c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 2-chamber apical view.
        a4c_diameters: Diameters measured at each key instant of the cardiac cycle, from the 4-chamber apical view.
        step_size:

    Returns:
        Left ventricle volume (in millilitres).
    """
    # All measures are now in millimeters, convert to meters by dividing by 1000
    a2c_diameters /= 1000
    a4c_diameters /= 1000
    step_size /= 1000

    # Estimate left ventricle volume from orthogonal disks
    lv_volume = np.sum(a2c_diameters * a4c_diameters) * step_size * np.pi / 4

    # Volume is now in cubic meters, so convert to milliliters (1 cubic meter = 1_000_000 milliliters)
    # return round(lv_volume * 1e6)
    return round(lv_volume * 1e6, 2) # no round

def _find_distance_to_edge(
    segmentation: np.ndarray, point_on_mid_line: np.ndarray, normal_direction: np.ndarray
) -> float:
    distance = 8  # start a bit in to avoid line stopping early at base
    while True:
        current_position = point_on_mid_line + distance * normal_direction

        y, x = np.round(current_position).astype(int)
        if segmentation.shape[0] <= y or y < 0 or segmentation.shape[1] <= x or x < 0:
            # out of bounds
            return distance

        elif segmentation[y, x] == 0:
            # Edge found
            return distance

        distance += 0.5

def _distance_line_to_points(line_point_0: np.ndarray, line_point_1: np.ndarray, points: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    return np.absolute(np.cross(line_point_1 - line_point_0, line_point_0 - points)) / np.linalg.norm(
        line_point_1 - line_point_0
    )

def _get_angle_of_lines_to_point(reference_point: np.ndarray, moving_points: np.ndarray) -> np.ndarray:
    diff = moving_points - reference_point
    return abs(np.degrees(np.arctan2(diff[:, 0], diff[:, 1])))

def _compute_diameters(segmentation: np.ndarray, voxelspacing: Tuple[float, float]) -> Tuple[np.ndarray, float]:
    """

    Args:
        segmentation: Binary segmentation of the structure for which to find the diameter.
        voxelspacing: Size of the segmentations' voxels along each (height, width) dimension (in mm).

    Returns:
    """

    # Make image isotropic, have same spacing in both directions.
    # The spacing can be multiplied by the diameter directly.
    segmentation, isotropic_spacing = resize_image_to_isotropic(segmentation, voxelspacing)

    # Go through entire contour to find AV plane
    contour = find_contours(segmentation, 0.5)[0] 

    # For each pair of contour points
    # Check if angle is ok
    # If angle is ok, check that almost all other contour points are above the line
    # Or check that all points between are close to the line
    # If so, it is accepted, select the longest stretch
    best_length = 0
    best_i, best_j = 0, 1
    for point_idx in range(2, len(contour)):# 需要至少一个中间点做“直线性检查”， 所以从2开始
        previous_points = contour[:point_idx]
        angles_to_previous_points = _get_angle_of_lines_to_point(contour[point_idx], previous_points)

        for acute_angle_idx in np.nonzero(angles_to_previous_points <= 45)[0]:
            intermediate_points = contour[acute_angle_idx + 1 : point_idx]
            distance_to_intermediate_points = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate_points <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx

    mid_point = int(best_j + round((best_i - best_j) / 2))
    # Apex is longest from midpoint
    mid_line_length = 0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    direction = contour[apex] - contour[mid_point]
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)  # Normalize
    diameters = []
    for fraction in np.linspace(0, 1, 20, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction

        distance1 = _find_distance_to_edge(segmentation, point_on_mid_line, normal_direction)
        distance2 = _find_distance_to_edge(segmentation, point_on_mid_line, -normal_direction)
        diameters.append((distance1 + distance2) * isotropic_spacing)

    step_size = (mid_line_length * isotropic_spacing) / 20
    return np.array(diameters), step_size

def _compute_diameters_debug(
    segmentation: np.ndarray,
    voxelspacing: Tuple[float, float],
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    和 _compute_diameters 一样，只是多了可视化：
    返回：
        - diameters: 20 个直径
        - step_size
        - img: 可视化图像 (H, W, 3) 的 numpy.uint8
    """

    # 1. 各向同性
    segmentation_iso, isotropic_spacing = resize_image_to_isotropic(segmentation, voxelspacing)

    # 2. 找轮廓
    contour = find_contours(segmentation_iso, 0.5)[0]
    
    # 3. 搜 AV 平面
    best_length = 0
    num_candidates = 0
    best_i, best_j = 0, 1
    distance_max = 0
    for point_idx in range(2, len(contour)):
        previous_points = contour[:point_idx]
        angles_to_previous_points = _get_angle_of_lines_to_point(contour[point_idx], previous_points)
        candidate_idxs = np.nonzero(angles_to_previous_points <= 45)[0]
        num_candidates += len(candidate_idxs)
        # 1. 计算分割掩膜的全局方向 (Major Axis Orientation)
        props = regionprops(segmentation.astype(int))
        if len(props) == 0: return np.array([]), 0.0 # 空掩膜保护
        
        # orientation 是主轴与X轴夹角 (-pi/2 到 pi/2)
        lv_orientation = props[0].orientation 
        
        # 定义 AV plane 理想法向量方向 (应该平行于主轴)
        # 或者说 AV plane 线段本身应该垂直于主轴

        for acute_angle_idx in np.nonzero(angles_to_previous_points <= 45)[0]:  
            intermediate_points = contour[acute_angle_idx + 1: point_idx]
            if len(intermediate_points) == 0:continue
            distance_to_intermediate_points = _distance_line_to_points(
                contour[point_idx], contour[acute_angle_idx], intermediate_points
            )
            if np.all(distance_to_intermediate_points <= 8):
                distance = np.linalg.norm(contour[point_idx] - contour[acute_angle_idx])
                if best_length < distance:
                    best_length = distance
                    best_i = point_idx
                    best_j = acute_angle_idx
            
            # if np.all(distance_to_intermediate_points <= 8):
                
            #     p1 = contour[point_idx]
            #     p2 = contour[acute_angle_idx]
            #     distance = np.linalg.norm(p1 - p2)
                
            #     # 计算当前候选直线的角度
            #     line_vector = p1 - p2
            #     line_angle = np.arctan2(line_vector[0], line_vector[1]) # 注意坐标系 y,x
                
            #     # 计算直线与LV主轴的夹角差 (理想情况应该是 90度 / pi/2)
            #     angle_diff = abs(line_angle - lv_orientation)
            #     # 归一化到 0-90度
            #     while angle_diff > np.pi/2: angle_diff = abs(angle_diff - np.pi)
                
            #     # 惩罚项：如果直线平行于主轴（即 angle_diff 接近 0），即使它很长也不要选
            #     # 真正的 AV plane 应该接近垂直主轴 (angle_diff 接近 pi/2)
                
            #     # 这是一个简单的加权 Score：长度 * 角度因子
            #     # 如果夹角小于 45度 (pi/4)，这一项会很小，抑制侧壁被选中
            #     angle_factor = np.sin(angle_diff) 
                
            #     # 只有当它大致垂直时才考虑 (比如 > 30度)
            #     is_roughly_perpendicular = angle_diff > (30 * np.pi / 180) 
                
            #     if is_roughly_perpendicular:
            #         # 只有符合几何方向约束的才比较长度
            #         if best_length < distance:
            #             best_length = distance
            #             best_i = point_idx
            #             best_j = acute_angle_idx
    print("len(contour)", len(contour),"best_length: ",best_length, "distance_max:", distance_max)

    print("Total candidates:", num_candidates) 
    mid_point = int(best_j + round((best_i - best_j) / 2)) # 中点索引

    # 4. 找 apex
    mid_line_length = 0
    apex = 0
    for i in range(len(contour)):
        length = np.linalg.norm(contour[mid_point] - contour[i])
        if mid_line_length < length:
            mid_line_length = length
            apex = i

    direction = contour[apex] - contour[mid_point]
    normal_direction = np.array([-direction[1], direction[0]])
    normal_direction = normal_direction / np.linalg.norm(normal_direction)

    diameters = []
    sample_lines = []
    for fraction in np.linspace(0, 1, 20, endpoint=False):
        point_on_mid_line = contour[mid_point] + direction * fraction

        d1 = _find_distance_to_edge(segmentation_iso, point_on_mid_line, normal_direction)
        d2 = _find_distance_to_edge(segmentation_iso, point_on_mid_line, -normal_direction)

        diameters.append((d1 + d2) * isotropic_spacing)

        p1 = point_on_mid_line + d1 * normal_direction
        p2 = point_on_mid_line - d2 * normal_direction
        sample_lines.append((p1, p2))

    step_size = (mid_line_length * isotropic_spacing) / 20
    diameters = np.array(diameters)

 # ========== 画图，但不在这里保存，转成 numpy 返回 ==========
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(segmentation_iso, cmap="gray")

    # 轮廓线
    ax.plot(contour[:, 1], contour[:, 0], 'g-', linewidth=1)

    # 轮廓点 (cyan)
    ax.scatter(contour[:, 1], contour[:, 0], c='cyan', s=15)

    # # 给每个 contour 点写 index（yellow）
    # for idx, (y, x) in enumerate(contour):
    #     ax.text(x, y, str(idx), color='yellow', fontsize=7)

    # AV plane
    ax.plot([contour[best_i, 1], contour[best_j, 1]],
            [contour[best_i, 0], contour[best_j, 0]],
            'r-', linewidth=2, label="AV plane")

    # Midline
    ax.plot([contour[mid_point, 1], contour[apex, 1]],
            [contour[mid_point, 0], contour[apex, 0]],
            'b-', linewidth=2, label="Midline")

    # 采样线
    for p1, p2 in sample_lines:
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'y-', linewidth=0.8)

    ax.scatter(contour[mid_point, 1], contour[mid_point, 0], c='cyan', s=30, label="mid_point")
    ax.scatter(contour[apex, 1], contour[apex, 0], c='magenta', s=30, label="apex")

    ax.set_title("LV contour & sampling lines")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=6)
    plt.tight_layout()

    # 转 numpy
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape((h, w, 3))

    plt.close(fig)

    return diameters, step_size, img
