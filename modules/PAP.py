import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PAP(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_prompt_tokens=4,
        num_freqs=2,
        scale_bound=0.2,
        shift_bound=0.05,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.num_freqs = num_freqs
        self.scale_bound = scale_bound
        self.shift_bound = shift_bound

        # shared phase-neutral LV prompt prior
        self.base_prompt = nn.Parameter(
            torch.randn(1, num_prompt_tokens, embed_dim) * 0.02
        )

        freqs = torch.arange(1, num_freqs + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)

        phase_dim = 3 + 2 * num_freqs

        # predict FiLM scale and shift
        self.phase_mlp = nn.Sequential(
            nn.Linear(phase_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2 * num_prompt_tokens * embed_dim)
        )

        # zero-init: training starts exactly from base_prompt
        nn.init.zeros_(self.phase_mlp[-1].weight)
        nn.init.zeros_(self.phase_mlp[-1].bias)

        # normalize base prompt before multiplicative modulation
        self.prompt_norm = nn.LayerNorm(embed_dim)

    def _normalize_phase_input(self, current_phase):
        if not torch.is_tensor(current_phase):
            raise TypeError("current_phase must be a torch.Tensor")

        if current_phase.dim() == 0:
            current_phase = current_phase.view(1, 1)
        elif current_phase.dim() == 1:
            current_phase = current_phase.unsqueeze(-1)
        elif current_phase.dim() != 2 or current_phase.shape[-1] != 1:
            raise ValueError(f"bad shape: {tuple(current_phase.shape)}")

        return current_phase

    def encode_phase(self, current_phase):
        phi = self._normalize_phase_input(current_phase)
        phi = phi.to(device=self.base_prompt.device, dtype=self.base_prompt.dtype)
        phi = phi.clamp(0.0, 1.0)

        poly_feat = torch.cat([phi, phi ** 2, phi ** 3], dim=-1)

        freqs = self.freqs.to(device=phi.device, dtype=phi.dtype).view(1, -1)
        angles = phi * freqs

        harmonic_feat = torch.cat(
            [torch.sin(angles), torch.cos(angles)],
            dim=-1
        )

        phase_feat = torch.cat([poly_feat, harmonic_feat], dim=-1)
        return phase_feat, phi

    def make_prompt_embeddings(self, current_phase):
        phase_feat, phi = self.encode_phase(current_phase)
        B = phase_feat.shape[0]

        base = self.base_prompt.expand(B, -1, -1)

        film_params = self.phase_mlp(phase_feat).view(
            B, self.num_prompt_tokens, 2, self.embed_dim
        )

        scale_raw = film_params[:, :, 0, :]
        shift_raw = film_params[:, :, 1, :]

        # bounded residual FiLM parameters
        delta_scale = self.scale_bound * torch.tanh(scale_raw)
        delta_shift = self.shift_bound * torch.tanh(shift_raw)

        # residual FiLM modulation
        base_norm = self.prompt_norm(base)
        prompt = base + delta_scale * base_norm + delta_shift

        return prompt, phase_feat