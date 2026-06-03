import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Optional, List

# from sam2.modeling.sam2_base import SAM2Base as Sam
from sam2.modeling.sam2echo_base import SAM2echo_Base as Sam


# --- Helper Classes (Modified for Dropout) ---

class _LoRA_qkv_Hiera(nn.Module):
    """Image Encoder 专用 LoRA：只对 Q/V 做增量"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        lora_dropout: float = 0.0, # [Added]
        scaling: float = 1.0,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.in_dim = qkv.in_features
        self.out_dim = qkv.out_features // 3
        
        # [Added] Dropout layer
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        
        # [Modified] Apply dropout to input of LoRA branch
        x_dropped = self.lora_dropout(x)
        
        new_q = self.linear_b_q(self.linear_a_q(x_dropped))
        new_v = self.linear_b_v(self.linear_a_v(x_dropped))

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q + self.scaling * new_q
        v = v + self.scaling * new_v
        return torch.cat([q, k, v], dim=-1)


class _LoRA_qkv_proj(nn.Module):
    """通用 LoRA Wrapper，支持 Linear 和 Conv2d"""

    def __init__(
        self,
        proj: nn.Module,
        w_a: nn.Module,
        w_b: nn.Module,
        lora_dropout: float = 0.0,
        scaling: float = 1.0,
    ):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b
        # [Added] Dropout layer
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.scaling = scaling

    def forward(self, x):
        # [Modified] Apply dropout to input of LoRA branch: B(A(dropout(x)))
        return self.proj(x) + self.scaling * self.w_b(self.w_a(self.lora_dropout(x)))


# --- LoRA Adapter (Modified) ---

class SamLoRAAdapter(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
        r: int,
        lora_dropout: float = 0.1, # [Added] Default dropout 0.05
        lora_layer: Optional[List[int]] = None,
        lora_alpha: Optional[float] = None,
        enable_image_encoder_lora: bool = True,
        enable_mask_decoder_lora: bool = True,
        enable_memory_attn_lora: bool = True,
        enable_memory_encoder_lora: bool = True,
    ):
        super().__init__()
        assert r > 0

        self.sam = sam_model
        self.sam.eval()
        self.r = r
        self.lora_dropout = lora_dropout # [Added]
        self.lora_alpha = float(r if lora_alpha is None else lora_alpha)
        if self.lora_alpha < 0:
            raise ValueError("lora_alpha must be non-negative")
        self.lora_scaling = self.lora_alpha / float(r)

        # -------- 1. Freeze 原模型参数 --------
        for p in self.sam.parameters():
            p.requires_grad_(False)
        # for p in self.sam.memory_encoder.parameters():
        #     p.requires_grad_(True)
        # for p in self.sam.memory_attention.parameters():
        #     p.requires_grad_(True)
        # -------- 2. 容器（ModuleList） --------
        # Image Encoder
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        # Mask Decoder
        self.self_attn_As = nn.ModuleList()
        self.self_attn_Bs = nn.ModuleList()
        self.cross_attn_ti_As = nn.ModuleList() 
        self.cross_attn_ti_Bs = nn.ModuleList() 
        self.cross_attn_it_As = nn.ModuleList()
        self.cross_attn_it_Bs = nn.ModuleList()

        # Memory Attention
        self.mem_attn_As = nn.ModuleList()
        self.mem_attn_Bs = nn.ModuleList()

        # Memory Encoder
        self.mem_enc_As = nn.ModuleList()
        self.mem_enc_Bs = nn.ModuleList()

        # Final Attn (Mask Decoder)
        self.fa_ti_q_proj_A = None
        self.fa_ti_q_proj_B = None
        self.fa_ti_v_proj_A = None
        self.fa_ti_v_proj_B = None

        # -------- 3. 注入 LoRA --------
        trunk = self.sam.image_encoder.trunk
        blocks = trunk.blocks
        self.lora_layer = list(lora_layer) if lora_layer is not None else list(range(len(blocks)))
        # trunk = self.sam.image_encoder.trunk
        # blocks = trunk.blocks
        # self.lora_layer = self._get_hiera_stage_block_ids(trunk, last_n_stages=2)
        if enable_image_encoder_lora:
            self._inject_image_encoder_lora(r, blocks)
        
        if enable_mask_decoder_lora and hasattr(self.sam, "sam_mask_decoder"):
            self._inject_decoder_lora(r)

        if enable_memory_attn_lora and hasattr(self.sam, "memory_attention"):
            self._inject_memory_attention_lora(r)

        if enable_memory_encoder_lora and hasattr(self.sam, "memory_encoder"):
            self._inject_memory_encoder_lora(r)

        # -------- 4. 初始化 LoRA 权重 --------
        self.reset_parameters()

    # ------------------- 通用 Helper ------------------- #

    @staticmethod
    def _make_linear_lora(in_dim: int, out_dim: int, r: int):
        """创建一对 Linear LoRA A/B 层"""
        w_a = nn.Linear(in_dim, r, bias=False)
        w_b = nn.Linear(r, out_dim, bias=False)
        return w_a, w_b

    @staticmethod
    def _make_conv1x1_lora(in_dim: int, out_dim: int, r: int):
        """创建一对 Conv2d(1x1) LoRA A/B 层"""
        w_a = nn.Conv2d(in_dim, r, kernel_size=1, bias=False)
        w_b = nn.Conv2d(r, out_dim, kernel_size=1, bias=False)
        return w_a, w_b

    def _replace_proj(self, module, attr_name, list_A, list_B, r: int):
        """替换 Linear 为带 LoRA 的 _LoRA_qkv_proj"""
        if not hasattr(module, attr_name):
            return
        linear_layer = getattr(module, attr_name)
        if isinstance(linear_layer, _LoRA_qkv_proj):
            return

        in_dim = linear_layer.in_features
        out_dim = linear_layer.out_features
        w_a, w_b = self._make_linear_lora(in_dim, out_dim, r)

        list_A.append(w_a)
        list_B.append(w_b)
        
        # [Modified] Pass lora_dropout
        setattr(
            module,
            attr_name,
            _LoRA_qkv_proj(
                linear_layer,
                w_a,
                w_b,
                lora_dropout=self.lora_dropout,
                scaling=self.lora_scaling,
            ),
        )

    def _replace_conv2d_proj(self, module, attr_name, list_A, list_B, r: int):
        """替换 1x1 Conv2d 为带 LoRA 的 _LoRA_qkv_proj"""
        if not hasattr(module, attr_name):
            return
        conv_layer = getattr(module, attr_name)
        if isinstance(conv_layer, _LoRA_qkv_proj):
            return

        if not isinstance(conv_layer, nn.Conv2d) or conv_layer.kernel_size != (1, 1):
            return

        in_dim = conv_layer.in_channels
        out_dim = conv_layer.out_channels
        w_a, w_b = self._make_conv1x1_lora(in_dim, out_dim, r)

        list_A.append(w_a)
        list_B.append(w_b)

        # [Modified] Pass lora_dropout
        setattr(
            module,
            attr_name,
            _LoRA_qkv_proj(
                conv_layer,
                w_a,
                w_b,
                lora_dropout=self.lora_dropout,
                scaling=self.lora_scaling,
            ),
        )

    # ------------------- 注入 Image Encoder LoRA ------------------- #
    def _get_hiera_stage_block_ids(self, trunk, last_n_stages: int = 2):
        """
        根据 Hiera 的 stage_ends，返回最后 n 个 stage 对应的 block id 列表
        例如:
        stage_ends = [0, 2, 9, 11]
        4个stage分别是:
            stage0: [0]
            stage1: [1,2]
            stage2: [3,4,5,6,7,8,9]
            stage3: [10,11]
        """
        if not hasattr(trunk, "stage_ends"):
            raise ValueError("trunk 没有 stage_ends，无法按 stage 选择 LoRA 层")

        stage_ends = list(trunk.stage_ends)   # e.g. [0, 2, 9, 11]
        num_stages = len(stage_ends)

        assert 1 <= last_n_stages <= num_stages

        # 先还原每个 stage 的 [start, end]
        stage_ranges = []
        start = 0
        for end in stage_ends:
            stage_ranges.append((start, end))
            start = end + 1

        # 取最后 n 个 stage
        selected_ranges = stage_ranges[-last_n_stages:]

        # 展开成 block id
        block_ids = []
        for s, e in selected_ranges:
            block_ids.extend(list(range(s, e + 1)))

        return block_ids
    def _inject_image_encoder_lora(self, r: int, blocks: nn.ModuleList):
        for i, blk in enumerate(blocks):
            if i not in self.lora_layer:
                continue
            attn = getattr(blk, "attn", None)
            if attn is None or not hasattr(attn, "qkv"):
                continue

            w_qkv_linear = attn.qkv
            in_dim = w_qkv_linear.in_features
            out_dim = w_qkv_linear.out_features // 3

            w_a_q, w_b_q = self._make_linear_lora(in_dim, out_dim, r)
            w_a_v, w_b_v = self._make_linear_lora(in_dim, out_dim, r)

            self.w_As += [w_a_q, w_a_v]
            self.w_Bs += [w_b_q, w_b_v]

            # [Modified] Pass lora_dropout
            attn.qkv = _LoRA_qkv_Hiera(
                w_qkv_linear,
                w_a_q,
                w_b_q,
                w_a_v,
                w_b_v,
                lora_dropout=self.lora_dropout,
                scaling=self.lora_scaling,
            )

    # ------------------- 注入 Mask Decoder LoRA ------------------- #
    def _inject_decoder_lora(self, r: int):
        decoder_transformer = self.sam.sam_mask_decoder.transformer

        for blk in decoder_transformer.layers:
            # self-attn
            self._replace_proj(blk.self_attn, "q_proj", self.self_attn_As, self.self_attn_Bs, r)
            self._replace_proj(blk.self_attn, "v_proj", self.self_attn_As, self.self_attn_Bs, r)
            # token -> image
            self._replace_proj(
                blk.cross_attn_token_to_image, "q_proj", self.cross_attn_ti_As, self.cross_attn_ti_Bs, r
            )
            self._replace_proj(
                blk.cross_attn_token_to_image, "v_proj", self.cross_attn_ti_As, self.cross_attn_ti_Bs, r
            )
            # image -> token
            self._replace_proj(
                blk.cross_attn_image_to_token, "q_proj", self.cross_attn_it_As, self.cross_attn_it_Bs, r
            )
            self._replace_proj(
                blk.cross_attn_image_to_token, "v_proj", self.cross_attn_it_As, self.cross_attn_it_Bs, r
            )

        # final_attn_token_to_image
        block = decoder_transformer.final_attn_token_to_image
        in_dim, out_dim = block.embedding_dim, block.internal_dim

        self.fa_ti_q_proj_A, self.fa_ti_q_proj_B = self._make_linear_lora(in_dim, out_dim, r)
        self.fa_ti_v_proj_A, self.fa_ti_v_proj_B = self._make_linear_lora(in_dim, out_dim, r)

        # [Modified] Pass lora_dropout
        block.q_proj = _LoRA_qkv_proj(
            block.q_proj,
            self.fa_ti_q_proj_A,
            self.fa_ti_q_proj_B,
            lora_dropout=self.lora_dropout,
            scaling=self.lora_scaling,
        )
        block.v_proj = _LoRA_qkv_proj(
            block.v_proj,
            self.fa_ti_v_proj_A,
            self.fa_ti_v_proj_B,
            lora_dropout=self.lora_dropout,
            scaling=self.lora_scaling,
        )
        
    # ------------------- 注入 Memory Attention LoRA ------------------- #

    def _inject_memory_attention_lora(self, r: int):
        memory_attention = self.sam.memory_attention
        for layer in memory_attention.layers:
            if hasattr(layer, "self_attn"):
                self._replace_proj(layer.self_attn, "q_proj", self.mem_attn_As, self.mem_attn_Bs, r)
                self._replace_proj(layer.self_attn, "v_proj", self.mem_attn_As, self.mem_attn_Bs, r)
            if hasattr(layer, "cross_attn_image"):
                self._replace_proj(layer.cross_attn_image, "q_proj", self.mem_attn_As, self.mem_attn_Bs, r)
                self._replace_proj(layer.cross_attn_image, "v_proj", self.mem_attn_As, self.mem_attn_Bs, r)

    # ------------------- 注入 Memory Encoder LoRA ------------------- #

    def _inject_memory_encoder_lora(self, r: int):
        """
        MemoryEncoder 结构：
          - pix_feat_proj: Conv2d 1x1
          - fuser: CXBlocks (pwconv1/2: Linear)
          - out_proj: Conv2d 1x1 或 Identity
        """
        mem_enc = self.sam.memory_encoder

        # 1. Pixel Feature Projection
        self._replace_conv2d_proj(mem_enc, "pix_feat_proj", self.mem_enc_As, self.mem_enc_Bs, r)

        # 2. Output Projection（非 Identity 才替换）
        if not isinstance(mem_enc.out_proj, nn.Identity):
            self._replace_conv2d_proj(mem_enc, "out_proj", self.mem_enc_As, self.mem_enc_Bs, r)

        # 3. Fuser CXBlocks
        if hasattr(mem_enc, "fuser") and hasattr(mem_enc.fuser, "layers"):
            for layer in mem_enc.fuser.layers:
                self._replace_proj(layer, "pwconv1", self.mem_enc_As, self.mem_enc_Bs, r)
                self._replace_proj(layer, "pwconv2", self.mem_enc_As, self.mem_enc_Bs, r)

    # ------------------- Init & Save/Load (保持不变) ------------------- #

    def reset_parameters(self) -> None:
        """Kaiming 初始化 A，B 全 0"""
        all_As = [
            self.w_As,
            self.self_attn_As,
            self.cross_attn_ti_As,
            self.cross_attn_it_As,
            self.mem_attn_As,
            self.mem_enc_As,
        ]
        all_Bs = [
            self.w_Bs,
            self.self_attn_Bs,
            self.cross_attn_ti_Bs,
            self.cross_attn_it_Bs,
            self.mem_attn_Bs,
            self.mem_enc_Bs,
        ]

        for w_list in all_As:
            for w in w_list:
                nn.init.kaiming_uniform_(w.weight, a=math.sqrt(5))

        for w_list in all_Bs:
            for w in w_list:
                nn.init.zeros_(w.weight)

        if self.fa_ti_q_proj_A is not None:
            nn.init.kaiming_uniform_(self.fa_ti_q_proj_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fa_ti_q_proj_B.weight)
            nn.init.kaiming_uniform_(self.fa_ti_v_proj_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fa_ti_v_proj_B.weight)

    # --- 保存 / 加载 只保存 LoRA + 少部分 sam 参数 --- #

    def save_lora_parameters(self, filename: str) -> None:
        state_dict = {}

        def save_list(prefix: str, module_list: nn.ModuleList):
            for i, layer in enumerate(module_list):
                state_dict[f"{prefix}_{i:03d}"] = layer.weight

        # image encoder
        save_list("w_a", self.w_As)
        save_list("w_b", self.w_Bs)

        # mask decoder
        save_list("sa_a", self.self_attn_As)
        save_list("sa_b", self.self_attn_Bs)
        save_list("cti_a", self.cross_attn_ti_As)
        save_list("cti_b", self.cross_attn_ti_Bs)
        save_list("cit_a", self.cross_attn_it_As)
        save_list("cit_b", self.cross_attn_it_Bs)

        # memory
        save_list("mem_attn_a", self.mem_attn_As)
        save_list("mem_attn_b", self.mem_attn_Bs)
        save_list("mem_enc_a", self.mem_enc_As)
        save_list("mem_enc_b", self.mem_enc_Bs)

        # final attn
        if self.fa_ti_q_proj_A is not None:
            state_dict["fati_qa"] = self.fa_ti_q_proj_A.weight
            state_dict["fati_qb"] = self.fa_ti_q_proj_B.weight
            state_dict["fati_va"] = self.fa_ti_v_proj_A.weight
            state_dict["fati_vb"] = self.fa_ti_v_proj_B.weight

        # 额外保存 prompt_encoder + mask_decoder(非 transformer 部分)
        full_state = self.sam.state_dict()
        for k, v in full_state.items():
            if "prompt_encoder" in k or ("mask_decoder" in k and "transformer" not in k):
                state_dict[k] = v

        torch.save(state_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location="cpu")

        def load_list(prefix: str, module_list: nn.ModuleList):
            for i, layer in enumerate(module_list):
                key = f"{prefix}_{i:03d}"
                if key in state_dict:
                    # 检查形状是否匹配，防止报错
                    saved_shape = state_dict[key].shape
                    current_shape = layer.weight.shape
                    if saved_shape != current_shape:
                        print(f"Warning: Shape mismatch for {key}. Saved: {saved_shape}, Current: {current_shape}. Skipping.")
                        continue
                    
                    # 使用 copy_ 将数据复制到现有设备上的 Tensor
                    with torch.no_grad():
                        layer.weight.data.copy_(state_dict[key])

        # image encoder
        load_list("w_a", self.w_As)
        load_list("w_b", self.w_Bs)

        # mask decoder
        load_list("sa_a", self.self_attn_As)
        load_list("sa_b", self.self_attn_Bs)
        load_list("cti_a", self.cross_attn_ti_As)
        load_list("cti_b", self.cross_attn_ti_Bs)
        load_list("cit_a", self.cross_attn_it_As)
        load_list("cit_b", self.cross_attn_it_Bs)

        # memory
        load_list("mem_attn_a", self.mem_attn_As)
        load_list("mem_attn_b", self.mem_attn_Bs)
        load_list("mem_enc_a", self.mem_enc_As)
        load_list("mem_enc_b", self.mem_enc_Bs)

        # final attn
        if "fati_qa" in state_dict and self.fa_ti_q_proj_A is not None:
            self.fa_ti_q_proj_A.weight = Parameter(state_dict["fati_qa"])
            self.fa_ti_q_proj_B.weight = Parameter(state_dict["fati_qb"])
            self.fa_ti_v_proj_A.weight = Parameter(state_dict["fati_va"])
            self.fa_ti_v_proj_B.weight = Parameter(state_dict["fati_vb"])

        # sam 其它部分（prompt_encoder / mask_decoder 头部）
        sam_state = self.sam.state_dict()
        to_update = {k: state_dict[k] for k in sam_state.keys() if k in state_dict}
        self.sam.load_state_dict(to_update, strict=False)
