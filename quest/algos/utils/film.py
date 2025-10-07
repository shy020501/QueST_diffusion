# quest/utils/film.py
from __future__ import annotations
from typing import Literal, Optional

import torch
from torch import nn
import torch.nn.functional as F


class ContextPooler(nn.Module):
    """
    context: (B, C_ctx, D) -> (B, D)
    mode="mean": 단순 평균 풀링
    mode="attn": 학습 가능한 1개 쿼리로 MHA attention pooling
    """
    def __init__(self, d_model: int, mode: Literal["mean", "attn"] = "mean", attn_heads: int = 1):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.query = nn.Parameter(torch.randn(1, 1, d_model))
            self.attn = nn.MultiheadAttention(d_model, num_heads=attn_heads, batch_first=True)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, C_ctx, D)
        if context.dim() != 3:
            raise ValueError(f"[ContextPooler] context must be (B, C_ctx, D), got {tuple(context.shape)}")

        if self.mode == "mean" or context.size(1) == 1:
            return context.mean(dim=1)

        if self.mode == "attn":
            B = context.size(0)
            q = self.query.expand(B, -1, -1)            # (B,1,D)
            y, _ = self.attn(q, context, context, need_weights=False)
            return y.squeeze(1)                          # (B,D)

        raise ValueError(f"[ContextPooler] unknown mode: {self.mode}")


class FiLMGen(nn.Module):
    """
    ctx_vec(+ t_vec) -> [gamma, beta]
    - per_layer=True: (B, n_layers, D)로 출력하여 레이어별로 다른 γ/β 적용
    - per_layer=False: (B, D)로 출력하여 인코더 전체에 단일 γ/β 적용
    - init_scale: γ 초기 범위를 좁히기 위해 tanh 후 스케일(초기 거의 identity 유지)
    """
    def __init__(
        self,
        ctx_dim: int,
        d_model: int,
        n_layers: int,
        per_layer: bool = True,
        use_t: bool = True,
        hidden_mult: int = 2,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.per_layer = per_layer
        self.use_t = use_t
        self.n_layers = n_layers
        self.d_model = d_model
        self.init_scale = init_scale

        in_dim = ctx_dim + (d_model if use_t else 0)
        out_dim = n_layers * d_model if per_layer else d_model
        hidden = max(d_model, hidden_mult * d_model)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * out_dim),  # [gamma | beta]
        )
        # 초기에는 거의 identity가 되도록 0으로 초기화
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ctx_vec: torch.Tensor, t_vec: Optional[torch.Tensor] = None):
        # ctx_vec: (B, D), t_vec: (B, D) or None
        if self.use_t and t_vec is not None:
            x = torch.cat([ctx_vec, t_vec], dim=-1)
        else:
            x = ctx_vec
        gamma_beta = self.net(x)                      # (B, 2*out_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = self.init_scale * torch.tanh(gamma)   # 안정화

        if self.per_layer:
            B = gamma.size(0)
            gamma = gamma.view(B, self.n_layers, self.d_model)
            beta  = beta.view(B,  self.n_layers, self.d_model)
        return gamma, beta


def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    x: (B, L, D)
    gamma/beta: (B, 1, D) 또는 (B, D) 또는 (B, L, D)
    """
    if gamma.dim() == 2:
        gamma = gamma.unsqueeze(1)  # (B,1,D)
    if beta.dim() == 2:
        beta = beta.unsqueeze(1)    # (B,1,D)
    return x * (1.0 + gamma) + beta
