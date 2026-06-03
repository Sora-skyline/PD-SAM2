from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from modules.lora_adapter import SamLoRAAdapter
from modules.IRR import IterativeLogitsRefiner
from sam2.modeling.sam2echo_base import SAM2echo_Base

from modules.PEG import PEG
from modules.PAP import PAP

class SAM2VideoTrainerWrapper(nn.Module):
    """
    Single-object, per-frame auto-prompt trainer wrapper with optional bidirectional pass.
    """

    def __init__(
        self,
        sam2_model: SAM2echo_Base,
        r: int,
        lora_layer=None,
        lora_alpha: Optional[float] = None,
        detach_memory: bool = False,
        use_phase: bool = True,
        use_memory: bool = True,
        use_refine: bool = True,
        use_external_sparse_prompt: bool = True,
    ):
        super().__init__()
        self.lora_adapter = SamLoRAAdapter(
            sam_model=sam2_model,
            r=r,
            lora_layer=lora_layer,
            lora_alpha=lora_alpha,
            enable_image_encoder_lora=True,
            enable_mask_decoder_lora=True,
            enable_memory_attn_lora=True,
            enable_memory_encoder_lora=True,   # 全参训练 memory
        )
        self.sam2 = self.lora_adapter.sam
        self.sam2.pred_obj_scores = False

        self.use_refine = use_refine
        self.mask_refine_module = IterativeLogitsRefiner(K=1) if use_refine else None
        self.auto_prompt_generator = PEG(num_queries=1)
        if use_external_sparse_prompt: 
            self.phase_prompt = PAP(embed_dim=256)

        self.detach_memory = detach_memory
        self.use_phase = use_phase
        self.use_memory = use_memory
        self.use_external_sparse_prompt = use_external_sparse_prompt

    def _resize_frames(self, videos: torch.Tensor):
        """Resize input videos to SAM2 image size if needed."""
        B, T, C, H, W = videos.shape
        target = self.sam2.image_size
        if (H, W) == (target, target):
            return videos, (H, W)
        videos_rs = F.interpolate(
            videos.view(B * T, C, H, W),
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, C, target, target)
        return videos_rs, (H, W)

    def extract_external_sparse_embeddings(
        self,
        frames: torch.Tensor,
        backbone_out: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Build PEG-based external sparse embeddings for a batch of frames.

        Args:
            frames: [B, C, H, W] input frames in image space.
            backbone_out: optional precomputed backbone outputs from `self.sam2.forward_image`.
        Returns:
            external_sparse_embeddings: [B, num_queries, 256]
        """
        if backbone_out is None:
            backbone_out = self.sam2.forward_image(frames)
        prompt_tokens = backbone_out["backbone_fpn"][-1].permute(0, 2, 3, 1).flatten(1, 2)
        external_sparse_embeddings, _ = self.auto_prompt_generator(prompt_tokens)
        return external_sparse_embeddings

    def forward(
        self,
        videos: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
        pts: Optional[tuple] = None,
    ):
        B, T, _, orig_h, orig_w = videos.shape
        device = videos.device
        phase_seq = None
        if self.use_phase and phases is not None:
            phase_seq = phases.to(device=device, dtype=videos.dtype)
            if phase_seq.dim() == 1:
                phase_seq = phase_seq.unsqueeze(0).expand(B, -1)
            elif phase_seq.dim() == 2:
                if phase_seq.shape[0] == 1 and B != 1:
                    phase_seq = phase_seq.expand(B, -1)
                elif phase_seq.shape[0] != B:
                    raise ValueError(
                        f"phases batch {phase_seq.shape[0]} does not match video batch {B}"
                    )
            else:
                raise ValueError(
                    f"phases must have shape [T] or [B, T], got {tuple(phase_seq.shape)}"
                )
            if phase_seq.shape[1] != T:
                raise ValueError(
                    f"phases length {phase_seq.shape[1]} does not match video length {T}"
                )

        videos_rs, orig_size = self._resize_frames(videos)

        # ---------------- Forward pass (single pass) ----------------
        output_dict_fw = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        pred_lowres_fw = [None] * T
        sam_feats_level0 = [None] * T if self.use_refine else None
        sam_feats_level1 = [None] * T if self.use_refine else None
        sam_feats_level2 = [None] * T if self.use_refine else None
        run_mem_encoder = self.use_memory
        fallback_phases = torch.linspace(
            0.0, 1.0, steps=T, device=device, dtype=videos.dtype
        ).unsqueeze(0).expand(B, -1)
        prompt_phases = phase_seq if phase_seq is not None else fallback_phases
        frame_phases = phase_seq[0] if phase_seq is not None else None

        def _run_one_frame(t: int, is_cond: bool) -> None:
            backbone_out_full = self.sam2.forward_image(videos_rs[:, t])
            # backbone_out_full["backbone_fpn"] = self.feature_enhance(backbone_out_full["backbone_fpn"])
            current_phi = prompt_phases[:, t].view(B, 1)

            if self.use_external_sparse_prompt:
                external_sparse_embeddings, _ = self.phase_prompt.make_prompt_embeddings(
                    current_phase=current_phi
                )
                point_inputs = None
            else:
                external_sparse_embeddings = None
                if pts is None:
                    raise ValueError(
                        "use_external_sparse_prompt=False requires random point prompts in pts."
                    )
                pt, point_labels = pts
                point_coords_t = pt[:, t]
                point_labels_t = point_labels[:, t]
                if point_coords_t.numel() == 0 or point_labels_t.numel() == 0:
                    raise ValueError(
                        "use_external_sparse_prompt=False requires non-empty random point prompts."
                    )
                point_inputs = (point_coords_t, point_labels_t)

            _, vision_feats, vision_pos_embeds, feat_sizes = self.sam2._prepare_backbone_features(
                backbone_out_full
            )

            current_out = self.sam2.track_step(
                frame_idx=t,
                is_init_cond_frame=is_cond,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,   # 如果你想启用点提示，这里改成 point_inputs
                mask_inputs=None,
                output_dict=output_dict_fw,
                num_frames=T,
                track_in_reverse=False,
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=None,
                external_sparse_embeddings=external_sparse_embeddings,
                external_dense_embeddings=None,
                frame_phases=frame_phases,
                use_memory=self.use_memory,
                phase_phi=current_phi,
                return_memory_conditioned_features=False,
            )

            if self.detach_memory and current_out.get("maskmem_features") is not None:
                current_out["maskmem_features"] = current_out["maskmem_features"].detach()

            key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            output_dict_fw[key][t] = current_out

            pred_lowres_fw[t] = current_out["pred_masks"]
            if self.use_refine:
                sam_feats_level0[t] = backbone_out_full["backbone_fpn"][0]
                sam_feats_level1[t] = backbone_out_full["backbone_fpn"][1]
                sam_feats_level2[t] = backbone_out_full["backbone_fpn"][2]

        cond_indices = [0]
        for t in range(T):
            _run_one_frame(t=t, is_cond=(t in cond_indices))

        pred_fw = torch.stack(pred_lowres_fw, dim=1)  # (B,T,1,h,w)
        pred_fw_up = F.interpolate(
            pred_fw.squeeze(2),  # (B,T,h,w)
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(2)  # (B,T,1,H,W)

        if not self.use_refine:
            return pred_fw_up

        BTHW = B * T
        images_for_refine = videos.view(BTHW, 3, orig_h, orig_w)
        coarse_for_refine = pred_fw_up.view(BTHW, 1, orig_h, orig_w)
        sam_feats_list = [
            torch.stack(sam_feats_level0, dim=1).reshape(BTHW, *sam_feats_level0[0].shape[1:]),
            torch.stack(sam_feats_level1, dim=1).reshape(BTHW, *sam_feats_level1[0].shape[1:]),
            torch.stack(sam_feats_level2, dim=1).reshape(BTHW, *sam_feats_level2[0].shape[1:]),
        ]

        refined_pred = self.mask_refine_module(
            images_for_refine, coarse_for_refine, sam_feats_list
        )

        pred = refined_pred.view(B, T, 1, orig_h, orig_w)
        return pred
