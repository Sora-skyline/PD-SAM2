import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
from utils.tools import draw_sem_seg_by_cv2_sum
from typing import Optional

MEMSAM_COMPARE_PALETTE = [[255, 255, 255], [37, 143, 36], [178, 48, 0], [178, 151, 0]]


def _to_chw_uint8_image(image):
    if not isinstance(image, np.ndarray):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.copy()

    image = image.astype(np.float32)
    if image.size > 0 and image.max() <= 1.0 + 1e-6:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/images', 'test', image_filename))
    img_ori = cv2.resize(img_ori, (256, 256))
    img_ori0 = cv2.imread(os.path.join(opt.data_path +'/images', 'test', image_filename))
    img_ori0 = cv2.resize(img_ori0, (256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    # table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
    #                           [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    table = np.array([[0, 0, 0], [0, 0, 155]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) 
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
          
    fulldir = opt.result_path + "/vis/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_npy(
    pred,
    gt,
    image_filename,
    opt,
    img_ori,
    frameidx: int,
    patient_name: Optional[str] = None,
    mask_logits: Optional[np.ndarray] = None,
) -> None:
    img_ori = img_ori[0,...]
    img_ori = _to_chw_uint8_image(img_ori)
    gt = (np.asarray(gt) > 0).astype(np.uint8)
    pred = (np.asarray(pred) > 0).astype(np.uint8)

    img = draw_sem_seg_by_cv2_sum(img_ori, gt, pred, MEMSAM_COMPARE_PALETTE)

    img = cv2.cvtColor(img.transpose(1,2,0), cv2.COLOR_RGB2BGR)
    img_ori = cv2.cvtColor(img_ori.transpose(1,2,0), cv2.COLOR_RGB2BGR)

    patient_dir = patient_name if patient_name is not None else image_filename.split(".")[0]
    base_dir = os.path.join(opt.result_path + "vis/", opt.modelname, patient_dir)
    pred_dir = os.path.join(base_dir, "Pred")
    gt_dir = os.path.join(base_dir, "GT")
    logits_dir = os.path.join(base_dir, "Logits")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    if mask_logits is not None:
        os.makedirs(logits_dir, exist_ok=True)

    img_stem = image_filename.split('.')[0] + f'_{frameidx}'
    cv2.imwrite(os.path.join(pred_dir, img_stem + ".png"), img)
    cv2.imwrite(os.path.join(gt_dir, img_stem + "_origin.png"), img_ori)

    if mask_logits is not None:
        if not isinstance(mask_logits, np.ndarray):
            mask_logits = mask_logits.detach().cpu().numpy()
        mask_logits = np.squeeze(mask_logits)
        prob_map = 1.0 / (1.0 + np.exp(-mask_logits))
        prob_uint8 = np.clip(prob_map * 255, 0, 255).astype(np.uint8)
        logits_vis = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(logits_dir, img_stem + "_logits.png"), logits_vis)
        # np.save(os.path.join(logits_dir, img_stem + "_logits.npy"), mask_logits)

def visual_segmentation_sets(seg, image_filename, opt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    #img = img_ori0
          
    fulldir = opt.result_path + "/" + opt.modelname + "/"
    #fulldir = opt.result_path + "/" + "GT" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    #img = img_ori0
    
    pt = np.array(pt.cpu())
    N = pt.shape[0]
    # for i in range(N):
    #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 6, (0,0,0), -1)
    #     cv2.circle(img, (int(pt[i, 0]), int(pt[i, 1])), 5, (0,0,255), -1)
    #     cv2.line(img, (int(pt[i, 0]-3), int(pt[i, 1])), (int(pt[i, 0])+3, int(pt[i, 1])), (0, 0, 0), 1)
    #     cv2.line(img, (int(pt[i, 0]), int(pt[i, 1])-3), (int(pt[i, 0]), int(pt[i, 1])+3), (0, 0, 0), 1)
          
    fulldir = opt.result_path + "/PT10-" + opt.modelname + "/"
    #fulldir = opt.result_path + "/PT3-" + "img" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)
