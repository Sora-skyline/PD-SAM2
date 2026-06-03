import os
# 请根据实际情况设置 CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys
print(sys.executable)

import argparse
from pickle import FALSE, TRUE
from statistics import mode
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config

# 引入 SAM 2 视频相关模块
from sam2.build_sam import  build_sam2
from sam2_trainer import SAM2VideoTrainerWrapper
# 引入 Utils
from utils.data_us import EchoVideoDataset, CamusDataset
from modules.augment import  JointTransform2DVideoTensor
from modules.losses import FixedScheduledSegLossWrapper
from modules.evaluate import eval_camus_test
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR


def str2bool(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("1", "true", "yes", "y")


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

def compute_sequence_loss(pred: torch.Tensor, gt: torch.Tensor, criterion, semi: bool, epoch: int):
    """
    pred/gt: [B, T, 1, H, W]
    """
    B, T, _, H, W = pred.shape
    if semi:
        # 仅计算首尾两帧的 Loss
        pred = pred[:,[0,-1]]
        gt = gt[:,[0,-1]]
    loss, raw_losses, weight_dict = criterion(pred, gt, current_epoch=epoch)
    return loss, raw_losses, weight_dict

def main():
    # ================================================= parameters setting =================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='default_all5_1', type=str, help='type of model')
    # parser.add_argument('--modelname', default='random_point', type=str, help='type of model') 
    # parser.add_argument('--modelname', default='test', type=str, help='type of model')
    parser.add_argument('--task', default='CAMUS_Video_Semi', help='task or dataset name')
    # parser.add_argument('--task', default='EchoNet', help='task or dataset name')

    # SAM 2 配置
    parser.add_argument('--model_ckpt', type=str, default='checkpoints/sam2.1_hiera_large.pt', help='Pretrained checkpoint of SAM 2')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l_echo.yaml', help='Config of SAM 2')
    # parser.add_argument('--model_ckpt', type=str, default='checkpoints/sam2.1_hiera_small.pt', help='Pretrained checkpoint of SAM 2')
    # parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_s_echo.yaml', help='Config of SAM 2')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu ')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of steps to accumulate gradients')

    parser.add_argument('--warmup',type=bool, default=True, help='warm up the learning')
    parser.add_argument('--warmup_period', type=int, default=250, help='warm up iterations')
    parser.add_argument('--keep_log', type=bool, default=True, help='keep the loss&lr&dice during training')
    parser.add_argument('--trainable_weights', type=bool, default=False, help='whether to train the uncertainty weights')
    parser.add_argument('--epoch',default=100, type=int, help='total epoch number to train')
    # 可控参数
    parser.add_argument('--use_phase', default=True, help='use phase/time encoding for temporal embeddings')
    parser.add_argument('--use_refine', default=True, type=str2bool, help='enable IterativeLogitsRefiner for ablation')
    parser.add_argument('--use_external_sparse_prompt', default=True, type=str2bool,
                        help='if true, inject PhasePrompt external_sparse_embeddings into track_step; if false, use random point prompts')

    # 视频特有参数
    parser.add_argument('--frame_length', type=int, default=10, help='number of frames per video clip')
    parser.add_argument('--eval_frame_length', type=int, default=10,
                        help='number of loaded clip frames used for val/test metrics; default is 10-frame Dice')
    parser.add_argument('--semi', default=True, type=str2bool, help='semi-supervised mode')
    parser.add_argument('--mode', choices=["image", "video"], default="video", help='unified mode selection')
    args = parser.parse_args()
    print(args)
    opt = get_config(args.task)
    opt.semi = args.semi
    opt.modelname = args.modelname
    opt.batch_size = args.batch_size * args.n_gpu
    mode = args.mode
    if mode == "image":
        dataset_frame_len = 2
        use_memory = False
    else:
        dataset_frame_len = args.frame_length
        use_memory = True
    args.metric_frame_mode = "clip"
    args.metric_frame_length = args.eval_frame_length
    device = torch.device(opt.device)
    
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("Using bfloat16 for training.")
        amp_dtype = torch.bfloat16
    else:
        print("Warning: bfloat16 not supported on this device. Falling back to float32 (or float16 if you implement Scaler).")
        amp_dtype = torch.float32 

    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)
    else:
        boardpath = None

    # ================================================== set random seed =============================================
    seed_value = 316
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ================================================== Data Loading ==================================================
    tf_train = JointTransform2DVideoTensor(
        img_size=opt.img_size,
        p_flip=0.0,
        p_rota=0.5,
        p_scale=0.5,
        p_gaussn=0.5,
        p_contr=0.5,
        p_gama=0.5,
        p_distor=0.0,
        color_jitter_params=(0.1,0.1,0.0,0.0),
        p_random_affine=0.0
    )
    # tf_train  = None
    tf_val = None
    tf_test = None

    if args.task == "EchoNet":
        dataset_cls = EchoVideoDataset
    elif args.task.startswith("CAMUS_"):
        dataset_cls = CamusDataset
    else:
        raise ValueError(f"Unsupported task for dataset selection: {args.task}")

    def build_dataset(split: str, joint_transform, endpoint_supervision_only: bool):
        return dataset_cls(
            opt.data_path,
            split,
            joint_transform,
            img_size=opt.img_size,
            frame_length=dataset_frame_len,
            endpoint_supervision_only=endpoint_supervision_only,
        )

    train_dataset = build_dataset(opt.train_split, tf_train, endpoint_supervision_only=opt.semi)
    val_dataset = build_dataset(opt.val_split, tf_val, endpoint_supervision_only=False)
    test_dataset = build_dataset(opt.test_split, tf_test, endpoint_supervision_only=False)
    trainloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # ================================================== build model ==================================================

    print(f"Building SAM 2 Video Predictor from {args.model_cfg}...")
    hydra_overrides = []
    sam2_base_model = build_sam2(
        config_file=args.model_cfg,
        ckpt_path=args.model_ckpt,
        device=device,
        hydra_overrides_extra=hydra_overrides,
    )

    model = SAM2VideoTrainerWrapper(
        sam2_base_model,
        4,
        use_phase=args.use_phase,
        use_memory=use_memory,
        use_refine=args.use_refine,
        use_external_sparse_prompt=args.use_external_sparse_prompt,
    ).to(device)
    model.to(device)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # ================================================== Loss & Optimizer ==================================================

    criterion = FixedScheduledSegLossWrapper(
        pos_weight=torch.tensor([1.0]).to(device), 
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=args.base_lr, 
                              betas=(0.9, 0.999), weight_decay=0.02)
    opt.epochs = args.epoch
    total_steps = opt.epochs * len(trainloader)
    warmup_steps = args.warmup_period if args.warmup else 0
    warmup_steps = min(warmup_steps, total_steps)

    # cosine: base_lr -> eta_min
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=0.00005, 
    )

    if warmup_steps > 0:
        # warmup: 从 1% base_lr -> 100% base_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_steps),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = cosine_scheduler

    # ================================================== Training Loop ==================================================
    iter_num = 0
    best_dice, loss_log = 0.0, np.zeros(opt.epochs+1)
    for epoch in range(opt.epochs):
            model.train()

            running_total_loss = 0.0
            running_raw_losses = {}

            epoch_pbar = tqdm(enumerate(trainloader), total=len(trainloader), ncols=130, 
                              desc=f"Epoch {epoch+1}/{opt.epochs}", leave=False, position=1)

            optimizer.zero_grad() 

            for batch_idx, (datapack) in epoch_pbar:
                imgs, masks, pts, phases = ensure_video_like(
                    datapack, device=opt.device, mode=mode, target_len=dataset_frame_len
                )

                # --- Forward (With AMP bf16) ---
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    logits = model(imgs, phases, pts)
                    logits = logits.float()

                    loss, raw_losses, weight_dict = compute_sequence_loss(
                        logits, masks, criterion, semi=opt.semi, epoch=epoch
                    )
                    
                    accum_loss = loss / args.accumulation_steps

                # --- Backward ---
                # 使用缩放后的 loss 进行反向传播
                accum_loss.backward()

                # 【核心修改 2】：达到步数或到达 DataLoader 末尾时更新参数
                if (batch_idx + 1) % args.accumulation_steps == 0 or (batch_idx + 1) == len(trainloader):
                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                iter_num += 1
                
                # --- Statistics ---
                running_total_loss += loss.item()
                
                for k, v in raw_losses.items():
                    if k not in running_raw_losses:
                        running_raw_losses[k] = 0.0
                    running_raw_losses[k] += v.item()
             
                current_lr = optimizer.param_groups[0]['lr']
                postfix_dict = {'loss': loss.item(), 'lr': current_lr}
                postfix_dict.update({k: v.item() for k, v in raw_losses.items()})
                epoch_pbar.set_postfix(postfix_dict)

            # End of Epoch Logging
            avg_loss = running_total_loss / len(trainloader)
            print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, avg_loss))
            loss_log[epoch] = avg_loss
            if args.keep_log:
                steps = max(len(trainloader), 1)
                TensorWriter.add_scalar('train/total_weighted_loss', avg_loss, epoch)
                for k, v in running_raw_losses.items():
                    TensorWriter.add_scalar(f'train/raw_{k}_loss', v / steps, epoch)
            torch.cuda.empty_cache()
            gc.collect() 
            # ================================================== Evaluation ==================================================
            if epoch % opt.eval_freq == 0:
                global_metrics = eval_camus_test(
                    valloader, model, criterion=criterion, opt=opt, args=args, epoch=epoch
                )
                mean_val_loss = global_metrics.get("loss", 0.0)
                mean_val_dice = global_metrics.get("dice", 0.0)
                val_raw_losses = global_metrics.get("raw_losses", {})
                steps = max(global_metrics.get("steps", 0), 1)
                
                print(f'epoch [{epoch}/{opt.epochs}], val loss:{mean_val_loss:.4f}, val dice:{mean_val_dice:.4f}')
                
                if args.keep_log:
                    TensorWriter.add_scalar('val/total_loss', mean_val_loss, epoch)
                    TensorWriter.add_scalar('val/dice', mean_val_dice, epoch)
                    for k, v in val_raw_losses.items():
                        TensorWriter.add_scalar(f'val/raw_{k}_loss', v / steps, epoch)

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice
                }
                # ================================================== Test ==================================================
                global_metrics = eval_camus_test(
                    testloader, model, criterion=criterion, opt=opt, args=args, epoch=epoch
                )
                mean_test_loss = global_metrics.get("loss", 0.0)
                mean_test_dice = global_metrics.get("dice", 0.0)
                test_raw_losses = global_metrics.get("raw_losses", {})
                steps = max(global_metrics.get("steps", 0), 1)

                print(f'epoch [{epoch}/{opt.epochs}], test loss:{mean_test_loss:.4f}, test dice:{mean_test_dice:.4f}')

                if args.keep_log:
                    TensorWriter.add_scalar('test/total_loss', mean_test_loss, epoch)
                    TensorWriter.add_scalar('test/dice', mean_test_dice, epoch)
                    for k, v in test_raw_losses.items():
                        TensorWriter.add_scalar(f'test/raw_{k}_loss', v / steps, epoch)
            # ================================================== Checkpoint Saving ==================================================
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice
                }
                if mean_val_dice > best_dice:
                    best_dice = mean_val_dice
                    if not os.path.isdir(opt.save_path):
                        os.makedirs(opt.save_path)
                    save_name = opt.save_path + args.modelname + opt.save_path_code + 'best.pth'
                    
                    torch.save(checkpoint, save_name)
                    print(f"Saved best model to {save_name}")
            

            if epoch % opt.save_freq == 0 or epoch == (opt.epochs-1):
                if not os.path.isdir(opt.save_path):
                    os.makedirs(opt.save_path)
                
                save_name = opt.save_path + args.modelname + opt.save_path_code + '_' + str(epoch) + '.pth'
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice
                }
                
                torch.save(checkpoint, save_name)
                print(f"Saved checkpoint to {save_name}")

import faulthandler; faulthandler.enable()
if __name__ == '__main__':
    main()
