# quest/algos/quest_modules/skill_d3pm.py
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from quest.algos.utils.film import FiLMGen, ContextPooler, apply_film


# ────────────────────────────────────────────────────────────────────────────────
# 작은 유틸: norm-first 트랜스포머 블록 (Self-Attn + [optional Cross-Attn] + FFN)
# ────────────────────────────────────────────────────────────────────────────────
class _Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_pdrop: float, ff_mult: int = 4,
                 use_cross_attn: bool = False):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_pdrop, batch_first=True)
        if use_cross_attn:
            self.ln_ca = nn.LayerNorm(d_model)
            self.ln_ctx = nn.LayerNorm(d_model)
            self.cross_attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_pdrop, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(attn_pdrop)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attn
        xs = self.ln1(x)
        sa, _ = self.self_attn(xs, xs, xs, need_weights=False)
        x = x + self.dropout(sa)

        # Cross-Attn (optional)
        if self.use_cross_attn:
            assert ctx is not None, "[_Block] cross-attn requires context"
            q = self.ln_ca(x)
            k = v = self.ln_ctx(ctx)
            ca, _ = self.cross_attn(q, k, v, need_weights=False)
            x = x + self.dropout(ca)

        # FFN
        xf = self.ln2(x)
        x = x + self.dropout(self.ff(xf))
        return x


class SkillD3PM(nn.Module):
    """
    D3PM denoiser with selectable conditioning:
      - cond_mode="film": Self-Attn + FFN, FiLM(γ/β) 주입
      - cond_mode="xattn": Self-Attn → Cross-Attn(context) → FFN
    Vocab = [0..codebook_size-1] + [MASK] (absorbing)
    """

    def __init__(
        self,
        *,
        vocab_size: int,          # = codebook_size + 1
        codebook_size: int,       # = J
        block_size: int,          # sequence length L
        n_layer: int,
        n_head: int,
        n_embd: int,
        attn_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        num_steps: int = 32,      # diffusion steps T

        # 누적 스케줄(ᾱ, γ̄)을 선형 보간 → per-step(α_t, γ_t) 역산
        alpha_bar_1: float = 0.99999,
        alpha_bar_T: float = 0.000009,
        gamma_bar_1: float = 0.000009,
        gamma_bar_T: float = 0.99999,

        # 조건 주입 모드: "film" | "xattn"
        cond_mode: str = "film",

        # FiLM 관련( cond_mode="film" 일 때만 사용 )
        film_per_layer: bool = True,
        film_use_time: bool = True,
        film_pool: str = "mean",   # "mean" | "attn"
        film_heads: int = 1,

        ctx_pos_type: str = "learned",   # "learned" | "sinusoidal"
        ctx_max_len: int = 64,           # context 길이 상한 (1 + T_obs 권장)

        device: str = "cuda",
    ):
        super().__init__()
        assert vocab_size == codebook_size + 1, \
            "[SkillD3PM] vocab_size must be codebook_size + 1 (mask)"
        self.vocab_size = int(vocab_size)
        self.codebook_size = int(codebook_size)
        self.mask_token = int(codebook_size)
        self.block_size = int(block_size)
        self.num_steps = int(num_steps)

        # ---- Embeddings ----
        self.n_embd = int(n_embd)
        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.pos_emb = nn.Embedding(self.block_size, self.n_embd)  # learned positional embedding (L fixed)
        self.t_emb = nn.Embedding(self.num_steps + 1, self.n_embd)  # t in [0..T]
        self.drop = nn.Dropout(embd_pdrop)

        # ---- 스케줄 (α_t, γ_t) ----
        with torch.no_grad():
            (alphas, gammas) = self._make_schedule(
                T=self.num_steps,
                alpha_bar_1=alpha_bar_1, alpha_bar_T=alpha_bar_T,
                gamma_bar_1=gamma_bar_1, gamma_bar_T=gamma_bar_T,
            )
        self.register_buffer("alphas", alphas.float())   # per-step α_t
        self.register_buffer("gammas", gammas.float())   # per-step γ_t

        # ---- 전이행렬 Q_t, 누적 \tilde{Q}_t ----
        q_one = []
        for t in range(self.num_steps):
            q_one.append(self._build_onestep_matrix(alpha=float(self.alphas[t]),
                                                    gamma=float(self.gammas[t])))
        q_one = torch.stack(q_one, dim=0)  # (T, V, V)
        self.register_buffer("q_onestep_mats", q_one.float())
        self.register_buffer("transpose_q_onestep_mats", torch.transpose(q_one, 1, 2))

        q_mats = [q_one[0]]
        for t in range(1, self.num_steps):
            q_mats.append(torch.matmul(q_mats[-1], q_one[t]))
        q_mats = torch.stack(q_mats, dim=0)  # (T, V, V)
        self.register_buffer("q_mats", q_mats.float())

        # ---- 조건 주입 모드 구성 ----
        self.cond_mode = cond_mode.lower()
        assert self.cond_mode in {"film", "xattn"}, f"Unknown cond_mode: {cond_mode}"
        
        self.ctx_pos_type = ctx_pos_type
        self.ctx_max_len = ctx_max_len
        if self.cond_mode == "xattn":
            if self.ctx_pos_type == "learned":
                self.ctx_pos_emb = nn.Embedding(ctx_max_len, self.n_embd)
            else:
                # 사인/코사인 포지션 버퍼
                pe = self._build_sinusoidal(ctx_max_len, self.n_embd)  # 아래 유틸 추가
                self.register_buffer("ctx_pos_sin", pe, persistent=False)

        self.layers = nn.ModuleList([
            _Block(
                d_model=self.n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                ff_mult=4,
                use_cross_attn=(self.cond_mode == "xattn"),
            )
            for _ in range(n_layer)
        ])
        self.final_ln = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.vocab_size)

        # ---- FiLM 구성 (film일 때만 사용) ----
        self.film_enabled = self.cond_mode == "film"
        if self.film_enabled:
            self.film_per_layer = film_per_layer
            self.film_use_time = film_use_time
            self.ctx_pooler = ContextPooler(self.n_embd, mode=film_pool, attn_heads=film_heads)
            self.film_gen = FiLMGen(
                ctx_dim=self.n_embd, d_model=self.n_embd, n_layers=n_layer,
                per_layer=film_per_layer, use_t=film_use_time, hidden_mult=2, init_scale=0.1
            )

        self.device_str = device

    # ───────────────────────────────────────────────────────────────────────
    # Codebook 크기 재설정(실제 FSQ 결과에 맞추어 재구성)
    # ───────────────────────────────────────────────────────────────────────
    def reconfigure_codebook_size(self, new_codebook_size: int):
        if int(new_codebook_size) == int(self.codebook_size):
            return
        new_vocab = int(new_codebook_size) + 1
        old_vocab = int(self.vocab_size)
        d_model   = self.n_embd
        dev       = self.device

        # Embedding resize
        old_emb = self.tok_emb
        new_emb = nn.Embedding(new_vocab, d_model).to(dev)
        with torch.no_grad():
            n_copy = min(old_vocab, new_vocab)
            new_emb.weight[:n_copy].copy_(old_emb.weight[:n_copy])
            if new_vocab > old_vocab:
                nn.init.normal_(new_emb.weight[n_copy:], mean=0.0, std=old_emb.weight.std().item())
        self.tok_emb = new_emb

        # Output head resize
        old_head = self.head
        new_head = nn.Linear(d_model, new_vocab, bias=True).to(dev)
        with torch.no_grad():
            n_copy = min(old_vocab, new_vocab)
            new_head.weight[:n_copy].copy_(old_head.weight[:n_copy])
            new_head.bias[:n_copy].copy_(old_head.bias[:n_copy])
            if new_vocab > old_vocab:
                nn.init.zeros_(new_head.bias[n_copy:])
                nn.init.normal_(new_head.weight[n_copy:], mean=0.0, std=old_head.weight.std().item())
        self.head = new_head

        # Update sizes
        self.codebook_size = int(new_codebook_size)
        self.vocab_size    = int(new_vocab)
        self.mask_token    = int(new_codebook_size)

        # Rebuild Q matrices
        q_one = []
        for t in range(self.num_steps):
            q_one.append(self._build_onestep_matrix(alpha=float(self.alphas[t]),
                                                    gamma=float(self.gammas[t])))
        q_one = torch.stack(q_one, dim=0).to(dev)
        for name in ["q_onestep_mats", "transpose_q_onestep_mats", "q_mats"]:
            if hasattr(self, name):
                delattr(self, name)
        self.register_buffer("q_onestep_mats", q_one)
        self.register_buffer("transpose_q_onestep_mats", torch.transpose(q_one, 1, 2))
        q_mats = [q_one[0]]
        for t in range(1, self.num_steps):
            q_mats.append(torch.matmul(q_mats[-1], q_one[t]))
        q_mats = torch.stack(q_mats, dim=0).to(dev)
        self.register_buffer("q_mats", q_mats)

    def _build_sinusoidal(self, length: int, dim: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)           # (L,1)
        i = torch.arange(dim, dtype=torch.float32).unsqueeze(0)                # (1,D)
        denom = torch.pow(10000, (2*(i//2))/dim)
        pe = pos / denom
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe  # (L, D)

    def _add_ctx_pos(self, context: torch.Tensor) -> torch.Tensor:
        # context: (B, C_ctx, D)
        B, C, D = context.shape
        idx = torch.arange(C, device=context.device)
        if self.ctx_pos_type == "learned":
            # 오버플로 안전: 길이가 최대치 넘으면 모듈러
            pos = self.ctx_pos_emb(idx % self.ctx_max_len)     # (C,D)
        else:
            pos = self.ctx_pos_sin[(idx % self.ctx_max_len)]   # (C,D)
        return context + pos.unsqueeze(0)                      # (B,C,D)

    # ───────────────────────────────────────────────────────────────────────
    # 누적 스케줄 → per-step 스케줄 (이름을 보다 일반적으로 변경)
    # ───────────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_schedule(
        T: int, alpha_bar_1: float, alpha_bar_T: float, gamma_bar_1: float, gamma_bar_T: float
    ):
        """
        alpha_bar:  \bar{α}_t 선형 보간 → α_t = alpha_bar[t] / alpha_bar[t-1]
        gamma_bar:  \bar{γ}_t 선형 보간 → γ_t = 1 - (1-γ̄_t)/(1-γ̄_{t-1})
        returns:
          alphas: (T,)
          gammas: (T,)
        """
        alpha_bar_raw = torch.linspace(alpha_bar_1, alpha_bar_T, steps=T)
        alpha_bar = torch.cat([torch.ones(1), alpha_bar_raw], dim=0)  # (T+1,)
        alpha = alpha_bar[1:] / alpha_bar[:-1]

        gamma_bar_raw = torch.linspace(gamma_bar_1, gamma_bar_T, steps=T)
        gamma_bar = torch.cat([torch.zeros(1), gamma_bar_raw], dim=0)  # (T+1,)
        one_minus_gamma_bar = 1.0 - gamma_bar
        one_minus_gamma = one_minus_gamma_bar[1:] / one_minus_gamma_bar[:-1]
        gamma = 1.0 - one_minus_gamma

        alpha = alpha.clamp(0.0, 1.0)
        gamma = gamma.clamp(0.0, 1.0)
        return alpha, gamma

    # ───────────────────────────────────────────────────────────────────────
    # One-step Q_t 생성
    # ───────────────────────────────────────────────────────────────────────
    def _build_onestep_matrix(self, alpha: float, gamma: float) -> torch.Tensor:
        V = self.vocab_size
        J = self.codebook_size
        M = self.mask_token
        Q = torch.zeros(V, V, dtype=torch.float32)

        residual = 1.0 - alpha - gamma
        tol = 1e-6
        if residual < -tol:
            raise ValueError(
                f"[SkillD3PM] alpha+gamma > 1 by {-(residual):.3e} "
                f"(alpha={alpha}, gamma={gamma}). Check schedule."
            )
        residual = max(residual, 0.0)
        beta = residual / (J - 1) if J > 1 else 0.0

        for i in range(J):
            Q[i, :J] = beta
            Q[i, i] = alpha
            Q[i, M] = gamma
        Q[M, M] = 1.0  # absorbing
        return Q

    # ───────────────────────────────────────────────────────────────────────
    # Forward (denoiser): (x_t, t, context) -> logits(x0)
    # ───────────────────────────────────────────────────────────────────────
    def forward(self, x_t: torch.LongTensor, t: torch.LongTensor, context: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        assert L == self.block_size, f"[SkillD3PM] expected L={self.block_size}, got {L}"

        # token + pos + time
        pos_idx = torch.arange(L, device=x_t.device)
        tok  = self.tok_emb(x_t)                                     # (B,L,D)
        pos  = self.pos_emb(pos_idx).unsqueeze(0)                    # (1,L,D)
        temb = self.t_emb(t.clamp(0, self.num_steps)).unsqueeze(1)   # (B,1,D)
        z    = tok + pos + temb                                      # (B,L,D)
        z    = self.drop(z)

        if self.cond_mode == "xattn":
            ctx = self._add_ctx_pos(context)   # ★ 여기서 컨텍스트에 포지션 주입
        else:
            ctx = context

        if self.cond_mode == "film" and self.film_enabled:
            ctx_vec = self.ctx_pooler(context)                       # (B,D)
            t_vec   = temb.squeeze(1) if self.film_use_time else None
            gamma, beta = self.film_gen(ctx_vec, t_vec)              # (B,n_layers,D) or (B,D)
        else:
            gamma = beta = None

        # 레이어 스택
        for li, layer in enumerate(self.layers):
            if self.cond_mode == "xattn":
                z = layer(z, ctx=ctx)   # Self → Cross → FFN
            else:
                z = layer(z, ctx=None)      # Self → FFN
                if gamma is not None and beta is not None:
                    if gamma.dim() == 3:  # per-layer
                        z = apply_film(z, gamma[:, li, :], beta[:, li, :])
                    else:
                        z = apply_film(z, gamma, beta)

        z = self.final_ln(z)
        logits = self.head(z)  # (B,L,V)
        return logits

    # ───────────────────────────────────────────────────────────────────────
    # q(x_t | x0), posterior helpers
    # ───────────────────────────────────────────────────────────────────────
    def _at_rows(self, A: torch.Tensor, t: torch.LongTensor, x: torch.LongTensor) -> torch.Tensor:
        B, L = x.shape
        idx_t = (t - 1).clamp_min(0)
        A_t = A.index_select(0, idx_t)                      # (B, V, V)
        x_onehot = F.one_hot(x.view(B, L), num_classes=A_t.size(1)).float()
        out = torch.matmul(x_onehot, A_t)                   # (B,L,V)
        return out

    def _at_rows_onehot(self, A: torch.Tensor, t: torch.LongTensor, x_oh: torch.Tensor) -> torch.Tensor:
        B, L, V = x_oh.shape
        idx_t = (t - 1).clamp_min(0)
        A_t = A.index_select(0, idx_t)                      # (B, V, V)
        out = torch.matmul(x_oh, A_t)                       # (B,L,V)
        return out

    @torch.no_grad()
    def q_sample(self, x0: torch.LongTensor, t: torch.LongTensor) -> torch.LongTensor:
        B, L = x0.shape
        probs = self._at_rows(self.q_mats, t, x0).clamp_min(1e-12)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        xt = torch.multinomial(probs.view(B * L, -1), 1).view(B, L)
        return xt

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits: bool):
        fact1 = torch.log(self._at_rows(self.transpose_q_onestep_mats, t, x_t).clamp_min(1e-12))
        if x_start_logits:
            x0_oh = F.softmax(x_start, dim=-1)  # (B,L,V)
            fact2 = torch.log(self._at_rows_onehot(self.q_mats, torch.where(t == 0, t, t - 1), x0_oh).clamp_min(1e-12))
            tzero_logits = x_start
        else:
            fact2 = torch.log(self._at_rows(self.q_mats, torch.where(t == 0, t, t - 1), x_start).clamp_min(1e-12))
            tzero_logits = torch.log(F.one_hot(x_start, num_classes=self.vocab_size).float().clamp_min(1e-12))
        out = fact1 + fact2
        t_b = t.view(-1, *([1] * (out.dim() - 1)))
        return torch.where(t_b == 0, tzero_logits, out)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, codebook_size: int) -> torch.LongTensor:
        B = context.size(0)
        L = self.block_size
        V = self.vocab_size
        M = self.mask_token

        x_t = torch.full((B, L), M, device=self.device, dtype=torch.long)
        for step in range(self.num_steps, 0, -1):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            logits_x0 = self.forward(x_t, t, context)                 # (B,L,V)
            post_logits = self.q_posterior_logits(
                x_start=logits_x0, x_t=x_t, t=t, x_start_logits=True
            )
            g = -torch.log(-torch.log(torch.rand_like(post_logits)))
            x_t = (post_logits + g).argmax(dim=-1)

        x0 = x_t
        if V > codebook_size:
            mask = (x0 >= codebook_size)
            if mask.any():
                with torch.no_grad():
                    logits_x0 = self.forward(
                        x0.clamp(max=V - 1),
                        torch.zeros(B, device=self.device, dtype=torch.long),
                        context,
                    )
                    real_logits = logits_x0[..., :codebook_size]
                    fill = real_logits.argmax(dim=-1)
                x0 = torch.where(mask, fill, x0)

        return x0.clamp_(min=0, max=codebook_size - 1)

    @property
    def device(self):
        return next(self.parameters()).device
