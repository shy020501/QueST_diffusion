import torch
import torch.nn.functional as F
import numpy as np
import quest.utils.tensor_utils as TensorUtils
import itertools
import math

from quest.algos.base import ChunkPolicy


class QueST_D3PM(ChunkPolicy):
    def __init__(self,
                 autoencoder,
                 policy_prior,
                 stage,
                 loss_fn,
                 l1_loss_scale,
                 lambda_x0_cfg=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.autoencoder = autoencoder
        self.policy_prior = policy_prior
        self.stage = stage

        self.l1_loss_scale = l1_loss_scale if stage == 2 else 0
        try:
            size = int(np.prod(self.autoencoder.fsq_level))
        except Exception:
            vq = getattr(self.autoencoder, "vq", None)
            if vq is None or not hasattr(vq, "codebook_size"):
                raise RuntimeError("[QueST_D3PM] cannot infer codebook size (fsq_level or vq.codebook_size).")
            size = int(vq.codebook_size)

        self.codebook_size = size

        if hasattr(self.policy_prior, "reconfigure_codebook_size"):
            self.policy_prior.reconfigure_codebook_size(self.codebook_size)

        self.loss = loss_fn

        self.lambda_x0_cfg = {
            "start": 1.0,
            "end": 0.1,
            "decay_type": "cosine",     # "cosine" | "linear"
            "decay_steps": None,        # None이면 set_schedule_from_epochs에서 채움
            "t_weighting": "none",      # "none" | "direct" | "inverse"
        }
        if lambda_x0_cfg is not None:
            self.lambda_x0_cfg.update(lambda_x0_cfg)

        self._global_step = 0
        self._decay_steps = self.lambda_x0_cfg.get("decay_steps", None)

    def set_schedule_from_epochs(self, *, num_epochs: int, steps_per_epoch: int):
        total = max(1, int(num_epochs) * int(steps_per_epoch))
        self._decay_steps = total
        # 사용자가 decay_steps를 명시했으면 그 값을 우선으로 사용
        if self.lambda_x0_cfg.get("decay_steps") is None:
            self.lambda_x0_cfg["decay_steps"] = total

    def _lambda_x0(self, t: torch.LongTensor) -> float:
        """스텝 기반 λ_x0 스케줄. (배치-스텝마다 갱신)"""
        start = float(self.lambda_x0_cfg.get("start", 1.0))
        end   = float(self.lambda_x0_cfg.get("end", 0.0))
        decay_type  = self.lambda_x0_cfg.get("decay_type", "cosine")
        decay_steps = self.lambda_x0_cfg.get("decay_steps") or self._decay_steps or 10000

        step = min(self._global_step, int(decay_steps))
        progress = step / float(decay_steps)

        if decay_type == "linear":
            lam = start + (end - start) * progress
        else:  # cosine
            lam = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))

        # 선택적으로 t 기반 가중
        mode = self.lambda_x0_cfg.get("t_weighting", "none")
        if mode != "none":
            # t는 [1..T], 평균 가중치를 스칼라로 곱해줌
            T = getattr(self.policy_prior, "num_steps", 1)
            t_norm = t.float().clamp_min(1.0) / float(T)
            if mode == "direct":       # 초기 t 작음 → 가중작음, 후반 t 큼 → 가중큼
                w = t_norm.mean().item()
            elif mode == "inverse":    # 초기 t 큼 → 가중큼, 후반 t 작음 → 가중작음
                w = (1.0 - t_norm).mean().item()
            else:
                w = 1.0
            lam *= w

        return float(lam)
        
    def get_optimizers(self):
        if self.stage == 0:
            decay, no_decay = TensorUtils.separate_no_decay(self.autoencoder)
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 1:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            optimizers = [
                self.optimizer_factory(params=decay),
                self.optimizer_factory(params=no_decay, weight_decay=0.)
            ]
            return optimizers
        elif self.stage == 2:
            decay, no_decay = TensorUtils.separate_no_decay(self, 
                                                            name_blacklist=('autoencoder',))
            decoder_decay, decoder_no_decay = TensorUtils.separate_no_decay(self.autoencoder.decoder)
            optimizers = [
                self.optimizer_factory(params=itertools.chain(decay, decoder_decay)),
                self.optimizer_factory(params=itertools.chain(no_decay, decoder_no_decay), weight_decay=0.)
            ]
            return optimizers

    def get_context(self, data):
        obs_emb = self.obs_encode(data)
        if obs_emb.dim() == 4: # (B,T,K,D) -> (B,T*K,D)
            B, T, K, D = obs_emb.shape
            obs_emb = obs_emb.view(B, T * K, D)
        task_emb = self.get_task_emb(data).unsqueeze(1)
        context = torch.cat([task_emb, obs_emb], dim=1)
        return context

    def compute_loss(self, data):
        if self.stage == 0:
            return self.compute_autoencoder_loss(data)
        elif self.stage == 1:
            return self.compute_prior_loss_d3pm(data)
        elif self.stage == 2:
            return self.compute_prior_loss_d3pm(data)

    def compute_autoencoder_loss(self, data):
        pred, pp, pp_sample, aux_loss, _ = self.autoencoder(data["actions"])
        recon_loss = self.loss(pred, data["actions"])
        if self.autoencoder.vq_type == 'vq':
            loss = recon_loss + aux_loss
        else:
            loss = recon_loss
            
        info = {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'aux_loss': aux_loss.sum().item(),
            'pp': pp.item(),
            'pp_sample': pp_sample.item(),
        }
        return loss, info

    def compute_prior_loss_d3pm(self, data):
        data = self.preprocess_input(data, train_mode=True)
        with torch.no_grad():
            x0 = self.autoencoder.get_indices(data["actions"]).long()

        context = self.get_context(data)
        B = x0.size(0)
        device = self.device

        t = torch.randint(1, getattr(self.policy_prior, "num_steps", 1) + 1, (B,), device=device)
        x_t = self.policy_prior.q_sample(x0, t)

        # 1) true posterior q(x_{t-1} | x_t, x0)
        q_post_logits = self.policy_prior.q_posterior_logits(
            x_start=x0, x_t=x_t, t=t, x_start_logits=False
        )
        q_post = torch.softmax(q_post_logits, dim=-1)

        # 2) model posterior p_theta(x_{t-1} | x_t) via x0-pred
        logits_x0 = self.policy_prior(x_t, t, context)      # (B,L,V)
        p_post_logits = self.policy_prior.q_posterior_logits(
            x_start=logits_x0, x_t=x_t, t=t, x_start_logits=True
        )

        # 3) KL(q || p) up to const = soft-label CE
        log_p = torch.log_softmax(p_post_logits, dim=-1)
        post_kl = -(q_post * log_p).sum(dim=-1).mean()   # scalar

        # 4) optional: x0 CE (mask 제외)
        logits_for_x0 = logits_x0[..., : self.codebook_size]
        ce_x0 = F.cross_entropy(
            logits_for_x0.reshape(-1, logits_for_x0.size(-1)),
            x0.view(-1)
        )

        lam = self._lambda_x0(t)
        prior_loss = post_kl + lam * ce_x0

        with torch.no_grad():
            sampled_indices = self.policy_prior.sample(context, codebook_size=self.codebook_size)  # (B,L)

        pred_actions = self.autoencoder.decode_actions(sampled_indices)  # (B, skill_block, act_dim)
        l1_loss = self.loss(pred_actions, data["actions"])
        total_loss = prior_loss + self.l1_loss_scale * l1_loss
        
        self._global_step += 1

        info = {
            "loss": total_loss.item(),
            "post_kl": post_kl.item(),
            "ce_x0": ce_x0.item(),
            "lambda_x0": float(lam),
            'l1_loss': l1_loss.item(),
            "step": int(self._global_step),
        }
        return total_loss, info


    def sample_actions(self, data):
        data = self.preprocess_input(data, train_mode=False)
        context = self.get_context(data)
        sampled_indices = self.policy_prior.sample(context, codebook_size=self.codebook_size)
        pred_actions = self.autoencoder.decode_actions(sampled_indices)  # (B, L, A)
        pred_actions = pred_actions.permute(1, 0, 2)  # (L, B, A)
        if pred_actions.size(1) == 1:
            pred_actions = pred_actions[:, 0, :]      # (L, A)
        return pred_actions.detach().cpu().numpy()
