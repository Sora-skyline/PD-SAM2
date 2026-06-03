# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(
                transformer_dim, transformer_dim // 8, kernel_size=1, stride=1
            )
            self.conv_s1 = nn.Conv2d(
                transformer_dim, transformer_dim // 4, kernel_size=1, stride=1
            )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        # stability-based 动态回退：单mask不稳就用多mask里 IoU 最大那个
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        adapter=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          torch.Tensor: batched SAM token for mask output
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
            adapter=adapter,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        # Prepare output
        return masks, iou_pred, sam_tokens_out, object_score_logits
    def get_tokens(
        self,
        batchsize: int,
    ) -> torch.Tensor:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # 1 iou + 1 total mask + 3 sub mask = (5, 256)
        output_tokens = output_tokens.unsqueeze(0).expand(batchsize, -1, -1) # b 5 256
        return output_tokens
    
    def predict_masks(
        self,
        image_embeddings: torch.Tensor,          # [B, C, H, W]
        image_pe: torch.Tensor,                  # [1, C, H, W]
        sparse_prompt_embeddings: torch.Tensor,  # [B, Np, C]
        dense_prompt_embeddings: torch.Tensor,   # [B, C, H, W]
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        adapter = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # ===== 1) 构造输出 tokens =====
        # token 顺序非常关键：
        # 如果 pred_obj_scores=True:
        #   [obj_score_token, iou_token, mask_tokens(0..K-1)]
        # 否则:
        #   [iou_token, mask_tokens(0..K-1)]
        s = 0  # s 用来标记 iou_token 的索引偏移（有obj_score_token时要+1）
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)

        # expand 到 batch： [B, Ntokens, C]
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        # ===== 2) 把输出tokens和prompt tokens拼起来，作为 transformer 的 token 输入 =====
        # tokens: [B, Ntokens + Np, C]
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # ===== 3) 图像特征是否需要repeat =====
        # repeat_image=True 表示：同一张图可能有多个prompt要分别预测mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings

        # ===== 4) 加上 dense prompt（例如 mask prompt） =====
        # src: [B, C, H, W]
        if adapter is None:
            src = src + dense_prompt_embeddings
        else:
            src = adapter(src, dense_prompt_embeddings)# (2 256 64 64)
        # image_pe 按设计 batch=1，然后 repeat 到 B
        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # ===== 5) transformer 前向 =====
        # hs: token 输出 [B, Ntokens+Np, C]
        # src: 图像输出（通常是展平后再回来）— 这里返回后续会 reshape
        hs, src = self.transformer(src, pos_src, tokens)

        # iou_token_out：取 iou_token 对应位置
        iou_token_out = hs[:, s, :]  # [B, C]

        # mask_tokens_out：取 K 个 mask token 的输出
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]  # [B, K, C]

        # ===== 6) 把 transformer 的图像输出恢复成 [B,C,H,W] 并上采样 =====
        src = src.transpose(1, 2).view(b, c, h, w)

        if not self.use_high_res_features:
            # 标准：两次反卷积上采样（x4）
            upscaled_embedding = self.output_upscaling(src)  # [B, C/8, H*4, W*4] (大致)
        else:
            # 使用高分辨率特征 skip connection
            # output_upscaling = [dc1, ln1, act1, dc2, act2]
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features  # 两级特征
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        # ===== 7) Hypernetwork：每个 mask token 生成一组线性组合权重 =====
        # 每个 MLP: [C] -> [C/8]，输出作为 mask 的“动态卷积权重”
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])  # [B, C/8]
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, K, C/8]

        # ===== 8) 生成 masks： (B,K,C') @ (B,C',HW) -> (B,K,HW) -> (B,K,H,W) =====
        b, c, h, w = upscaled_embedding.shape  # 这里 c 应该就是 C/8
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        # masks: [B, K, H, W]，是 logits（未sigmoid/softmax）

        # ===== 9) 预测每个 mask 的质量（IoU评分） =====
        iou_pred = self.iou_prediction_head(iou_token_out)  # [B, K]

        # ===== 10) 物体存在性得分（可选） =====
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])  # [B,1]
        else:
            # 默认认为物体存在：logit=10 => sigmoid(10)≈1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits


    def _get_stability_scores(self, mask_logits):
        """
        Compute stability scores of the mask logits based on the IoU between upper and
        lower thresholds.
        """
        mask_logits = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits > stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        stability_scores = torch.where(area_u > 0, area_i / area_u, 1.0)
        return stability_scores

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """
        When outputting a single mask, if the stability score from the current single-mask
        output (based on output token 0) falls below a threshold, we instead select from
        multi-mask outputs (based on output token 1~3) the mask with the highest predicted
        IoU score. This is intended to ensure a valid mask for both clicking and tracking.
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(
            multimask_iou_scores.size(0), device=all_iou_scores.device
        )
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
