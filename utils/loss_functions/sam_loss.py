from pyexpat import model
import torch
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class DC_and_BCE_loss(nn.Module):
    def __init__(self, classes=2, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.ce =  CrossEntropyLoss()
        self.dc = DiceLoss(classes)
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 4:
            target = target[:, 0, :, :]
        loss_ce = self.ce(low_res_logits, target[:].long())
        loss_dice = self.dc(low_res_logits, target, softmax=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class MaskDiceLoss(nn.Module):
    def __init__(self):
        super(MaskDiceLoss, self).__init__()

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1)) # b h w -> b 1 h w
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, net_output, target, weight=None, sigmoid=False):
        if sigmoid:
            net_output = torch.sigmoid(net_output) # b 1 h w
        assert net_output.size() == target.size(), 'predict {} & target {} shape do not match'.format(net_output.size(), target.size())
        dice_loss = self._dice_loss(net_output[:, 0], target[:, 0])
        return dice_loss

class Mask_DC_and_BCE_loss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits']
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss

class Mask_DC_and_BCE_lossV2(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_lossV2, self).__init__()
        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice
        return loss
    
class SobelEdgeExtractor(nn.Module):
    """
    使用Sobel算子提取图像边缘的模块
    """
    def __init__(self, kernel_size=3):
        super(SobelEdgeExtractor, self).__init__()
        self.kernel_size = kernel_size
        
        # 定义水平和垂直Sobel核
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 将核注册为buffer，不参与梯度更新
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, x):
        # x: (B, 1, H, W) 概率图 (0-1之间)
        # 为了处理可能的边界问题，使用replicate padding

        x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
        device = x.device
        
        # 2. 确保卷积核也在同一个设备上
        # 虽然 register_buffer 通常会自动管理，但为了防止意外，这里强制对齐
        # 注意：expand 之前必须先 to(device)
        curr_sobel_x = self.sobel_x.to(device)
        curr_sobel_y = self.sobel_y.to(device)
        grad_x = F.conv2d(x_pad, curr_sobel_x)
        grad_y = F.conv2d(x_pad, curr_sobel_y)
        
        # 计算梯度幅值: sqrt(gx^2 + gy^2 + eps)
        edge = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 归一化边缘强度到 [0, 1] 区间，以便计算 BCE
        # 注意：Sobel响应可能大于1，Tanh或者Clip是一个好选择，这里简单用Clip
        edge = torch.clamp(edge, 0, 1)
        
        return edge

class Mask_DC_and_BCE_lossV2_edge(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8, edge_weight=0.1 , hd_weight=0.001, 
                 debug_save_path="debug_edges", log_interval=50):
        super(Mask_DC_and_BCE_lossV2_edge, self).__init__()

        # 1. 主任务 Loss (针对 Mask)
        self.ce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss() 

        # 2. 边缘提取器
        self.edge_extractor = SobelEdgeExtractor()

        self.dice_weight = dice_weight
        self.edge_weight = edge_weight
        self.hd_weight = hd_weight
        # --- 可视化相关初始化 ---
        self.debug_save_path = debug_save_path
        self.log_interval = log_interval
        self.counter = 0 # 内部计数器
        # 创建保存目录
        if not os.path.exists(self.debug_save_path):
            os.makedirs(self.debug_save_path)
    def get_thick_edge_gt(self, mask_gt):

        disk_kernel = torch.tensor([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]
        ], dtype=torch.float32, device=mask_gt.device)

        # 调整维度以适配 F.conv2d: (Out_channels, In_channels, H, W) -> (1, 1, 5, 5)
        disk_kernel = disk_kernel.unsqueeze(0).unsqueeze(0)

        # 2. 准备 Input
        # 确保输入是 float 类型 (B, 1, H, W)
        mask_tensor = mask_gt.float()

        # 3. 模拟膨胀 (Dilation)
        # 使用 conv2d 进行卷积，padding=2 保持尺寸不变
        # 只要卷积结果 > 0，说明该像素周围 (在 disk 范围内) 存在至少一个前景点
        with torch.no_grad(): # GT 生成不需要梯度
            conv_out = F.conv2d(mask_tensor, disk_kernel, padding=2)
            dilated = (conv_out > 0).float()

        # 4. 获取边缘 (Dilated - Original)
        edge = dilated - mask_tensor

        # 5. 确保数值稳定性 (只取 0 或 1)
        edge = torch.clamp(edge, 0, 1)

        return edge
    # def get_thick_edge_gt(self, mask_gt, kernel_size=5):
    #     padding = kernel_size // 2
    #     dilated = F.max_pool2d(mask_gt.float(), kernel_size=kernel_size, stride=1, padding=padding)
    #     # 2. 边缘 = 膨胀 - 原始
    #     edge = dilated - mask_gt.float()
    #     return edge # 返回 0~1 的宽边缘

    def edge_soft_dice_loss(self, pred_edge, gt_edge, smooth=1e-5):
        """
        专门为边缘设计的 Soft Dice Loss
        pred_edge: [B, 1, H, W] 范围 0-1
        gt_edge:   [B, 1, H, W] 范围 0-1
        """
        # 展平以便计算 (B, -1)
        pred_flat = pred_edge.view(pred_edge.size(0), -1)
        gt_flat = gt_edge.view(gt_edge.size(0), -1)

        intersection = (pred_flat * gt_flat).sum(1)
        union = pred_flat.sum(1) + gt_flat.sum(1)

        # Dice = 2*Inter / (Union + smooth)
        dice_score = (2. * intersection + smooth) / (union + smooth)

        # Loss = 1 - Dice
        return 1 - dice_score.mean()

    def forward(self, net_output, target):
        # 1. 维度处理
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            net_output = net_output.view(-1, net_output.shape[2], net_output.shape[3], net_output.shape[4])
        
        # 2. 主 Mask Loss (BCE + Dice)
        loss_ce = self.ce(net_output, target)
        loss_dice = self.dc(net_output, target, sigmoid=True)
        base_loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        # 3. 边缘 Loss (改用 Dice)

        pred_prob = torch.sigmoid(net_output)
        target_float = target.float()

        # 提取边缘 (0~1 的浮点数图)
        pred_edge = self.edge_extractor(pred_prob)
        gt_edge = self.get_thick_edge_gt(target) 

        if self.edge_weight > 0:    
            loss_edge = self.edge_soft_dice_loss(pred_edge, gt_edge)
        else:
            loss_edge = torch.tensor(0.0, device=net_output.device)
        if self.training and (self.counter % self.log_interval == 0):
            self.save_debug_images(pred_prob, pred_edge, gt_edge, target_float)
        
        self.counter += 1
        # 4. HD Loss (可选)
        loss_hd = self.hd_loss_func(net_output, target) if self.hd_weight > 0 else torch.tensor(0.0, device=net_output.device)

        dice_loss = loss_dice * self.dice_weight
        hd_loss = loss_hd * self.hd_weight 
        edge_loss = loss_edge * self.edge_weight

        # 5. 总 Loss
        total_loss = base_loss + edge_loss + hd_loss
        
        return total_loss, base_loss, edge_loss, hd_loss

    def save_debug_images(self, pred_prob, pred_edge, gt_edge, target):
        """
        将 Tensor 转为 Image 并保存对比图
        """
        # 取 Batch 中的第一张图 (index 0)
        # Tensor Shape: (B, 1, H, W) -> detech -> cpu -> numpy -> (H, W)
        with torch.no_grad():
            img_pred = pred_prob[0, 0].detach().cpu().numpy()
            img_pred_edge = pred_edge[0, 0].detach().cpu().numpy()
            img_gt = target[0, 0].detach().cpu().numpy()
            img_gt_edge = gt_edge[0, 0].detach().cpu().numpy()

        # 创建 matplotlib 画布
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # 1. 预测掩膜
        axes[0, 0].imshow(img_pred, cmap='gray')
        axes[0, 0].set_title("Pred Mask (Prob)")
        axes[0, 0].axis('off')

        # 2. GT 掩膜
        axes[0, 1].imshow(img_gt, cmap='gray')
        axes[0, 1].set_title("GT Mask")
        axes[0, 1].axis('off')

        # 3. 预测边缘
        axes[1, 0].imshow(img_pred_edge, cmap='magma') # magma 配色更容易看清边缘强度
        axes[1, 0].set_title(f"Pred Edge\n(Step {self.counter})")
        axes[1, 0].axis('off')

        # 4. GT 边缘
        axes[1, 1].imshow(img_gt_edge, cmap='magma')
        axes[1, 1].set_title("GT Edge")
        axes[1, 1].axis('off')

        plt.tight_layout()
        
        # 保存图片
        save_name = os.path.join(self.debug_save_path, f"vis_step_{self.counter:05d}.png")
        plt.savefig(save_name)
        plt.close(fig) # 重要！关闭画布防止内存泄漏
        # print(f"Saved debug image to {save_name}") # 可选：打印提示

class Mask_DC_and_BCE_lossV2_phase(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.8):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_DC_and_BCE_lossV2, self).__init__()
        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dc = MaskDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, net_output, target, pred_vec, gt_phase):
        low_res_logits = net_output
        if len(target.shape) == 5:
            target = target.view(-1, target.shape[2], target.shape[3], target.shape[4])
            low_res_logits = low_res_logits.view(-1, low_res_logits.shape[2], low_res_logits.shape[3], low_res_logits.shape[4])
        loss_ce = self.ce(low_res_logits, target)
        loss_dice = self.dc(low_res_logits, target, sigmoid=True)
        loss = (1 - self.dice_weight) * loss_ce + self.dice_weight * loss_dice

        gt_vec = torch.stack([torch.sin(gt_phase), torch.cos(gt_phase)], dim=-1)
        L_phase = F.mse_loss(pred_vec, gt_vec)

        return loss
    
class Mask_BCE_loss(nn.Module):
    def __init__(self, pos_weight):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!
        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Mask_BCE_loss, self).__init__()

        self.ce =  torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, net_output, target):
        low_res_logits = net_output['low_res_logits'] 
        loss = self.ce(low_res_logits, target)
        return loss

def get_criterion(modelname='SAM', opt=None):
    device = torch.device(opt.device)
    pos_weight = torch.ones([1]).cuda(device=device)*2
    if modelname == "SAMed":
        criterion = DC_and_BCE_loss(classes=opt.classes)
    elif modelname == "MSA":
        criterion = Mask_BCE_loss(pos_weight=pos_weight)
    elif modelname == "XMemSAM" or modelname == "MemSAM" or modelname == 'SAM2_lora' or modelname == 'SAM2_lora_auto' or modelname == 'SAM2_lora_auto_semi' or modelname == 'SAM2_lora_semi' or modelname == 'SAM2_Video_Train' or modelname == 'SAM2_Video_Test' or modelname == 'SAM2_Video_Train_auto':
        criterion = Mask_DC_and_BCE_lossV2(pos_weight=pos_weight)
    elif modelname == "SAM2_Video_Train_edge" or "SAM2_Video_Train_WOedge" or "SAM2_Video_Train_edge_hd":
        criterion = Mask_DC_and_BCE_lossV2_edge(pos_weight=pos_weight)
    elif modelname == 'SAM2_lora_auto_phase':
        criterion = Mask_DC_and_BCE_lossV2_phase(pos_weight=pos_weight)
    else:
        criterion = Mask_DC_and_BCE_loss(pos_weight=pos_weight)
    return criterion
