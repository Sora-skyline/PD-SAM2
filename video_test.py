import os

from modules.losses import FixedScheduledSegLossWrapper
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import csv
import json
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
# from modules.eval import eval_camus_test
from modules.evaluate import eval_camus_test
from utils.config import get_config
from utils.generate_prompts import get_click_prompt
from utils.data_us import EchoVideoDataset,CamusDataset

from sam2.build_sam import build_sam2
# from sam2_video_trainer_version2 import SAM2VideoTrainerWrapper
from sam2_trainer import SAM2VideoTrainerWrapper


def str2bool(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes", "y")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SAM2 Video Test / Evaluation")

    # parser.add_argument('--modelname', default='default_all5_1', type=str)
    parser.add_argument('--modelname', default='default_all5', type=str)

    parser.add_argument('--task', default='CAMUS_Video_Semi')
    # parser.add_argument('--task', default='EchoNet', help='task or dataset name')

    parser.add_argument('--model_ckpt', type=str, default='checkpoints/sam2.1_hiera_large.pt', help='Pretrained checkpoint of SAM 2')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l_echo.yaml', help='Config of SAM 2')

    parser.add_argument('--compute_ef', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_gpu', type=int, default=1)

    parser.add_argument('--frame_length', type=int, default=10,
                        help='number of frames per clip, -1 means use full sequence from dataset')
    parser.add_argument('--infer_all_frames', default=False, action='store_true',
                        help='load and save the full ED-to-ES sequence instead of the default clip length')
    parser.add_argument('--semi', default=False, type=str2bool,
                        help='if true, compute metrics on ED/ES only; if false, compute metrics on the configured clip frames')

    parser.add_argument('--use_phase',default=True,help='use phase/time encoding for temporal embeddings')
    parser.add_argument('--use_refine', default=True, type=str2bool,help='enable IterativeLogitsRefiner for ablation')
    parser.add_argument('--use_external_sparse_prompt', default=True, type=str2bool,help='if true, inject PhasePrompt external_sparse_embeddings into track_step; if false, use random point prompts')

    parser.add_argument('--mode', choices=["image", "video"], default="video", help='unified mode selection')
    parser.add_argument('--lora_ckpt', type=str,
                        default="checkpoints/default_all5_1_best.pth")
    return parser.parse_args()


def build_dataloader(args, opt):
    """
    构建 EchoVideoDataset 的 test dataloader
    """
    if args.task == "EchoNet":
        dataset_cls = EchoVideoDataset
    elif args.task.startswith("CAMUS_"):
        dataset_cls = CamusDataset

    else:
        raise ValueError(f"Unsupported task for dataset selection: {args.task}")

    def build_dataset(split: str, joint_transform):
        return dataset_cls(
            opt.data_path,
            split,
            joint_transform,
            frame_length=args.frame_length,
            endpoint_supervision_only=args.semi,
        )

    tf_test = None    # 测试一般不做 data augmentation
    test_dataset = build_dataset(opt.test_split, tf_test)
    batch_size = args.batch_size * args.n_gpu
    if args.infer_all_frames:
        batch_size = 1

    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return testloader


def load_lora_video_model(args, opt, device):
    """
    加载 SAM2 + LoRA 的视频模型
    """
    sam2_model = build_sam2(
        config_file=args.model_cfg,
        ckpt_path=args.model_ckpt,
        device=device,
    )

    # LoRA rank = 4，可以按需改
    model = SAM2VideoTrainerWrapper(
        sam2_model,
        r=4,
        use_phase=args.use_phase,
        use_memory=args.use_memory,
        use_refine=args.use_refine,
        use_external_sparse_prompt=args.use_external_sparse_prompt,
    )
    model.to(device)

    print(f"Loading LoRA checkpoint from: {args.lora_ckpt}")
    checkpoint = torch.load(args.lora_ckpt, map_location=device, weights_only=True)

    ckpt = checkpoint['model_state_dict']

    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        new_state_dict = {k[7:]: v for k, v in ckpt.items()}
    else:
        new_state_dict = ckpt

    model.load_state_dict(new_state_dict, strict=False)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    return model



def main():
    args = parse_args()
    args.metric_frame_length = args.frame_length if args.frame_length and args.frame_length > 0 else 10
    if args.infer_all_frames:
        args.frame_length = -1
    if args.frame_length <= 0:
        args.infer_all_frames = True
    print(args)

    opt = get_config(args.task)
    opt.semi = args.semi
    opt.batch_size = 1 if args.infer_all_frames else args.batch_size * args.n_gpu
    opt.mode = "test"
    opt.modelname = args.modelname
    if args.mode == "image":
        # 默认取首尾两帧（data_us.py 在 frame_length<=2 时即取首/尾）
        args.use_memory = False
    else:
        args.use_memory = True
    device = torch.device(opt.device)
    model = load_lora_video_model(args, opt, device)
    criterion = FixedScheduledSegLossWrapper(
        pos_weight=torch.tensor([1.0]).to(device),
        enable_edge=False,
    ).to(device)
    testloader = build_dataloader(args, opt)
    model.eval()
    profile_stats = None
    # profile_stats = collect_model_profile(model, args, opt, testloader, device)
    # print("\nModel Profiling")
    # print(f"Profile input shape: {profile_stats['profile_input_shape']}")
    # if profile_stats["macs_clip"] is not None:
    #     print(f"Model MACs/clip: {profile_stats['macs_clip']:.0f} ({_format_count(profile_stats['macs_clip'])}) [{profile_stats['flops_source']}]")
    #     print(f"Model FLOPs (Table-5 aligned, G/frame as GMAC): {profile_stats['gmacs_per_frame']:.3f}G")
    #     print(f"Model FLOPs (GFLOPs/frame, 2xMAC): {profile_stats['gflops_per_frame_2x']:.3f}G")
    # else:
    #     print(f"Model FLOPs: unavailable [{profile_stats['flops_source']}]")
    # print(f"Total params: {profile_stats['total_params']} ({_format_count(profile_stats['total_params'])})")
    # print(f"Trainable params: {profile_stats['trainable_params']} ({_format_count(profile_stats['trainable_params'])})")

    # ============ 进度条包装 dataloader ============
    print("\n🔍 Starting inference ...\n")
    pbar = tqdm(testloader, desc="Inference", ncols=120)

    # dice_mean, iou_mean, hd_mean, assd_mean, dices_std, iou_std, hd_std, assd_std = eval_camus_test(pbar, model, criterion=criterion, opt=opt, args=args)
    global_metrics = eval_camus_test(pbar, model, criterion=criterion, opt=opt, args=args, save_vis=True)
    print("dataset:" + args.task + " -----------model name: "+ args.modelname)
    print(f"frame_eval_mode: {global_metrics.get('frame_eval_mode', 'endpoints')}")
    print(f"frame_output_mode: {global_metrics.get('frame_output_mode', 'loaded_clip')}")
    print(f"metric_frame_count: {global_metrics.get('metric_frame_count', 0)}")
    print(f"output_frame_count: {global_metrics.get('output_frame_count', 0)}")
    print("dice_mean  iou_mean  hd_mean  assd_mean")
    print(global_metrics["dice"], global_metrics["iou"], global_metrics["hd95"], global_metrics["assd"], global_metrics["ef_corr"],global_metrics["ef_bias"],global_metrics["ef_std"])
    fps = float(global_metrics.get('fps', 0.0))
    print(f"FPS: {fps:.4f}")

    if profile_stats is not None:
        total_params_m = profile_stats["total_params"] / 1e6
        trainable_params_m = profile_stats["trainable_params"] / 1e6
        g_per_frame = profile_stats["gmacs_per_frame"] if profile_stats["gmacs_per_frame"] is not None else float("nan")
        print("\nTable-5 Aligned Summary")
        print("Model FLOPs(G/frame)  Total params(M)  Trainable params(M)  FPS")
        print(f"{g_per_frame:.3f}  {total_params_m:.3f}  {trainable_params_m:.3f}  {fps:.3f}")


if __name__ == "__main__":
    main()
