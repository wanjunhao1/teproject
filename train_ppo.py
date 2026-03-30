# train_ppo.py
from __future__ import annotations

import random
import logging
from dataclasses import dataclass
from typing import Optional, Dict
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training.log',
    filemode='w'
)

# Try to import TensorBoard, if not available, we'll skip logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    logging.info("Warning: TensorBoard not available. Training will continue without logging.")

from env import RealTEEnv
from real_data_loader import build_real_topology_and_paths


# -----------------------------
# Utilities

# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_torch(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# -----------------------------
# Masked Grouped Logistic-Normal on simplex
# -----------------------------
class MaskedGroupedLogisticNormal:
    """
    Grouped Logistic-Normal on simplex with per-group feasibility mask.

    Params:
      mu, log_std: (B, N, K) logits-space Normal params (independent across K dims)
      mask:        (B, N, K) bool, True means feasible

    Action:
      y: (B, N, K) simplex per group with infeasible entries exactly 0 (after renorm).

    log_prob:
      For each group, restrict to feasible subset of size m.
      Use anchor (last feasible) to map to R^{m-1}:
        u_i = log(y_i) - log(y_anchor)
      Given g ~ Normal(mu, diag(std^2)), u = g[:-1] - g_anchor
      => u ~ MVN(mu_u, Cov = diag(std[:-1]^2) + std_anchor^2 * 11^T
      log p(y) = log p(u) - sum_{i in feasible} log y_i
      Degenerate m=1 => log_prob = 0 (stable PPO convention).
    """

    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor, eps: float = 1e-8):
        assert mu.shape == log_std.shape
        self.mu = mu
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self.eps = eps

    @staticmethod
    def _project_simplex_with_mask(y: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
        y = y * mask.to(y.dtype)
        s = y.sum(dim=-1, keepdim=True).clamp_min(eps)
        return y / s

    def sample(self, mask: torch.Tensor) -> torch.Tensor:
        g = self.mu + self.std * torch.randn_like(self.mu)
        neg_inf = torch.finfo(g.dtype).min
        g = torch.where(mask, g, torch.full_like(g, neg_inf))
        y = torch.softmax(g, dim=-1)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return self._project_simplex_with_mask(y, mask, self.eps)

    def log_prob(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        y:    (B,N,K)
        mask: (B,N,K) bool
        returns: (B,N) individual log prob for each active group
        """
        y = torch.clamp(y, self.eps, 1.0)
        y = self._project_simplex_with_mask(y, mask, self.eps)

        B, N, K = y.shape
        device = y.device
        dtype = y.dtype

        # Find the last valid index for each (b, n)
        idx_grid = torch.arange(K, device=device).view(1, 1, K)
        masked_idx = torch.where(mask, idx_grid, torch.tensor(-1, device=device))
        last_idx = masked_idx.max(dim=-1, keepdim=True).values

        m_counts = mask.sum(dim=-1, keepdim=True)

        is_anchor = mask & (idx_grid == last_idx)
        is_other = mask & (~is_anchor)

        is_anchor_f = is_anchor.to(dtype)
        is_other_f = is_other.to(dtype)

        # Avoid -inf * 0 = nan by clamping before log
        y_safe = y.clamp_min(self.eps)
        log_y = torch.log(y_safe)
        
        log_y_anchor = (log_y * is_anchor_f).sum(dim=-1, keepdim=True)
        mu_anchor = (self.mu * is_anchor_f).sum(dim=-1, keepdim=True)
        var = self.std.pow(2)
        var_anchor = (var * is_anchor_f).sum(dim=-1, keepdim=True)

        diff = (log_y - log_y_anchor) - (self.mu - mu_anchor)
        diff = diff * is_other_f

        inv_var = 1.0 / var.clamp_min(self.eps)
        inv_var_masked = inv_var * is_other_f

        term1 = (diff.pow(2) * inv_var_masked).sum(dim=-1)
        u_Dinv_1 = (diff * inv_var_masked).sum(dim=-1)
        one_Dinv_1 = inv_var_masked.sum(dim=-1)

        s = var_anchor.squeeze(-1)
        denom = 1.0 + s * one_Dinv_1
        quad_form = term1 - (s * u_Dinv_1.pow(2)) / denom.clamp_min(self.eps)

        log_det_D = (torch.log(var.clamp_min(self.eps)) * is_other_f).sum(dim=-1)
        log_det = log_det_D + torch.log(denom.clamp_min(self.eps))

        m_minus_1 = (m_counts.squeeze(-1) - 1).clamp_min(0)
        log_p_u = -0.5 * (m_minus_1 * math.log(2 * math.pi) + log_det + quad_form)

        log_abs_det_jacobian = -(log_y * mask.to(dtype)).sum(dim=-1)
        group_log_prob = log_p_u + log_abs_det_jacobian

        m_vals = m_counts.squeeze(-1)
        group_log_prob = torch.where(m_vals <= 0, torch.tensor(0.0, device=device, dtype=dtype), group_log_prob)
        group_log_prob = torch.where(m_vals == 1, torch.tensor(0.0, device=device, dtype=dtype), group_log_prob)

        return group_log_prob

    def entropy_proxy(self, mask: torch.Tensor) -> torch.Tensor:
        """
        A stable entropy proxy in logit space.
        Returns per-group entropy proxy with shape (B, N).
        """
        var = self.std.pow(2)
        # Entropy of diagonal Normal in logits space: 0.5 * sum(log(2*pi*e*var))
        ent_per_dim = 0.5 * torch.log(2.0 * math.pi * math.e * var.clamp_min(self.eps))
        ent_group = (ent_per_dim * mask.to(ent_per_dim.dtype)).sum(dim=-1)

        # Groups with <=1 feasible path are effectively deterministic.
        valid_stochastic = (mask.sum(dim=-1) > 1).to(ent_group.dtype)
        return ent_group * valid_stochastic


# -----------------------------
# Actor-Critic with grouped masked policy
# -----------------------------
class ActorCriticGroupedLogisticNormal(nn.Module):
    def __init__(self, obs_dim: int, n_groups: int, k_paths: int, hidden: int = 256):
        super().__init__()
        self.N = n_groups
        self.K = k_paths
        self.action_dim = n_groups * k_paths  # Removed admission logic for routing stabilization

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, self.N * self.K)
        self.log_std_head = nn.Linear(hidden, self.N * self.K)
        self.value_head = nn.Linear(hidden, 1)

        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        # Start with low policy variance so training begins near smooth routing, not pure noise.
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -2.0)

        self.LOG_STD_MIN = -5.0
        self.LOG_STD_MAX = 0.5

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        mu = self.mu_head(x).view(-1, self.N, self.K)
        log_std = self.log_std_head(x).view(-1, self.N, self.K).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        value = self.value_head(x).squeeze(-1)
        return mu, log_std, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        mu, log_std, value = self.forward(obs)

        # Routing distribution
        dist = MaskedGroupedLogisticNormal(mu, log_std)
        routing_action = dist.sample(mask)                     # (B,N,K)
        routing_logprob = dist.log_prob(routing_action, mask)  # (B,N)

        B = mu.shape[0]
        # Combine actions into flat format [N*K]
        action_flat = routing_action.view(B, -1)
        return action_flat, routing_logprob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_flat: torch.Tensor, mask: torch.Tensor):
        mu, log_std, value = self.forward(obs)

        # Split actions
        routing_actions = actions_flat.view(-1, self.N, self.K)

        # Routing logprob
        dist = MaskedGroupedLogisticNormal(mu, log_std)
        routing_logprob = dist.log_prob(routing_actions, mask)

        # Use a stable entropy proxy in logits space.
        entropy_est = dist.entropy_proxy(mask)
        total_logprob = routing_logprob
        return total_logprob, entropy_est, value


# -----------------------------
# PPO Buffer + mask
# -----------------------------
class RolloutBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int, device: str, n_groups: int, k_paths: int):
        self.size = size
        self.device = device
        self.N = n_groups
        self.K = k_paths

        self.obs = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((size, action_dim), dtype=torch.float32, device=device)
        self.masks = torch.zeros((size, self.N, self.K), dtype=torch.bool, device=device)

        self.logprobs = torch.zeros((size, self.N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((size,), dtype=torch.float32, device=device)
        self.values = torch.zeros((size,), dtype=torch.float32, device=device)

        self.advantages = torch.zeros((size,), dtype=torch.float32, device=device)
        self.returns = torch.zeros((size,), dtype=torch.float32, device=device)

        self.ptr = 0

    def add(self, obs, action, mask, logprob, reward, done, value):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.masks[i] = mask
        self.logprobs[i] = logprob
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, lam: float):
        last_gae = 0.0
        for t in reversed(range(self.size)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std(unbiased=False) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get_minibatches(self, minibatch_size: int):
        idx = torch.randperm(self.size, device=self.device)
        for start in range(0, self.size, minibatch_size):
            mb = idx[start : start + minibatch_size]
            yield (
                self.obs[mb],
                self.actions[mb],
                self.masks[mb],
                self.logprobs[mb],
                self.advantages[mb],
                self.returns[mb],
            )


# -----------------------------
# PPO config
# -----------------------------
@dataclass
class PPOConfig:
    seed: int = 0
    device: str = "cpu"

    # env
    num_nodes: int = 6
    num_links: int = 10
    n_demands: int = 30  # Reduced N size. Action vector is now much denser and meaningful
    k_paths: int = 6
    episode_len: int = 60

    # training
    total_steps: int = 1500_000  # Expand total steps slightly
    rollout_steps: int = 480  # = 8 * episode_len (60), ensures trajectory completeness
    gamma: float = 0.99  # Standard value
    gae_lambda: float = 0.95  # Standard value
    lr_start: float = 2e-4  # Slightly higher LR to improve early learning speed
    lr_end: float = 1e-5    # End learning rate (linear decay)
    clip_range: float = 0.15  # Slightly tighter clipping to reduce policy drift
    value_coef: float = 0.5  # Keep value coefficient
    entropy_coef: float = 1e-4  # Keep exploration but avoid suppressing stronger policy signal
    max_grad_norm: float = 0.5  # Standard value
    update_epochs: int = 3  # Slightly more optimization per rollout for clearer policy improvement
    minibatch_size: int = 120  # More stable gradient estimate with rollout_steps=480
    log_every_updates: int = 1


def ppo_update(model, optimizer, buffer: RolloutBuffer, cfg: PPOConfig, entropy_coef: float):
    clip = cfg.clip_range
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy_loss = 0.0
    total_loss = 0.0
    
    for _ in range(cfg.update_epochs):
        for obs, actions, masks, old_logp, adv, returns in buffer.get_minibatches(cfg.minibatch_size):
            logp, entropy_est, values = model.evaluate_actions(obs, actions, masks)
            active_groups = (masks.sum(dim=-1) > 1).to(logp.dtype)
            active_per_sample = active_groups.sum(dim=-1).clamp_min(1.0)

            # Clamp log probabilities defensively to avoid massive exponentiation explosions
            diff = torch.clamp(logp - old_logp, min=-5.0, max=5.0)
            ratio = torch.exp(diff) # (B, N)

            surr1 = ratio * adv.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv.unsqueeze(-1)
            surr = torch.min(surr1, surr2)
            # Sum over groups to better reflect joint action improvement per environment step.
            policy_loss = -((surr * active_groups).sum(dim=-1)).mean()
            value_loss = 0.5 * (returns - values).pow(2).mean()
            entropy_bonus = ((entropy_est * active_groups).sum(dim=-1) / active_per_sample).mean()

            loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_bonus.item()
            total_loss += loss.item()
    
    # Calculate average losses
    num_updates = cfg.update_epochs * (cfg.rollout_steps // cfg.minibatch_size)
    avg_policy_loss = total_policy_loss / num_updates
    avg_value_loss = total_value_loss / num_updates
    avg_entropy_loss = total_entropy_loss / num_updates
    avg_total_loss = total_loss / num_updates
    
    return avg_policy_loss, avg_value_loss, avg_entropy_loss, avg_total_loss


def train(cfg: PPOConfig):
    logging.info("Starting train function...")
    set_seed(cfg.seed)
    device = cfg.device
    logging.info(f"Using device: {device}")

    # Create TensorBoard writer if available
    writer = SummaryWriter() if TENSORBOARD_AVAILABLE else None
    logging.info(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")

    # Use real topology instead of demo topology
    logging.info("Loading real topology...")
    links, candidate_paths = build_real_topology_and_paths(
        topology_name="abilene"  # You can change this to "geant" or "usnet"
    )

    # Get num_nodes from the topology
    num_nodes = max(max(u, v) for u, v, _, _ in links) + 1
    logging.info(f"Topology loaded with {num_nodes} nodes")
    logging.info(f"Number of links: {len(links)}")
    logging.info(f"Number of candidate paths: {len(candidate_paths)}")

    env = RealTEEnv(
        num_nodes=num_nodes,
        links=links,
        candidate_paths=candidate_paths,
        n_demands=cfg.n_demands,
        k_paths=cfg.k_paths,
        episode_len=cfg.episode_len,
        seed=cfg.seed,
        use_real_traffic=True,  # Use real traffic data
        traffic_trace="mixed",  # Use mixed traffic for better training
        # Dynamic features (Stage 1: static environment for stable routing learning)
        dynamic_capacity=False,  # Disable time-varying link capacity initially
        capacity_variation=0.0,  # 0% capacity variation
        link_failure_prob=0.0,  # No link failures
        traffic_burstiness=0.0,  # No burstiness
    )

    logging.info(f"Environment created with {len(links)} links")
    logging.info(f"Observation dimension: {env.obs_dim}")
    logging.info(f"Action dimension: {env.action_dim}")

    obs_dim = env.obs_dim
    N = env.num_demands_per_step
    K = env.k_paths
    action_dim = env.action_dim

    model = ActorCriticGroupedLogisticNormal(obs_dim, n_groups=N, k_paths=K, hidden=512).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr_start)

    logging.info("Model created")

    obs_np, _ = env.reset()
    logging.info(f"First observation shape: {obs_np.shape}")
    logging.info(f"Initial active flows: {len(env.active_flows)}")
    logging.info(f"Initial active coflows: {len(env.active_coflows)}")
    obs = to_torch(obs_np, device)

    n_updates = cfg.total_steps // cfg.rollout_steps
    global_step = 0

    # Track episode rewards and metrics
    episode_rewards = []
    current_episode_reward = 0.0
    
    # Track cumulative metrics per episode
    episode_deadline_miss = 0
    episode_deadline_total = 0
    episode_coflow_cct_sum = 0
    episode_coflow_count = 0

    logging.info(f"Starting training with {n_updates} updates")

    for update in range(1, n_updates + 1):
        # Linear learning rate decay
        frac = 1.0 - (update - 1.0) / n_updates
        lr_now = frac * cfg.lr_start + (1.0 - frac) * cfg.lr_end
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        logging.info(f"\n=== Update {update}/{n_updates} (LR: {lr_now:.2e}) ===")
        buffer = RolloutBuffer(obs_dim, action_dim, cfg.rollout_steps, device, n_groups=N, k_paths=K)
        last_info: Optional[Dict] = None
        
        # Accumulate metrics for smoothing
        total_mlu = 0.0
        total_deadline_penalty = 0.0
        total_coflow_cct = 0.0
        total_change_cost = 0.0
        total_infeas_mass = 0.0
        total_capacity_scale = 0.0
        total_adm_ratio_dl = 0.0
        total_adm_ratio_cf = 0.0
        total_adm_ratio_bk = 0.0
        total_reward_abs = 0.0
        total_reward_delta = 0.0
        total_reward_mlu_cost = 0.0
        total_reward_dl_miss_rate = 0.0
        total_reward_delay_penalty = 0.0
        total_reward_throughput_eff = 0.0
        num_steps = 0
        
        logging.info(f"  Initial active flows: {len(env.active_flows)}")
        logging.info(f"  Initial active coflows: {len(env.active_coflows)}")

        for _ in range(cfg.rollout_steps):
            global_step += 1

            mask_np = env.get_mask()  # (N,K) bool
            mask = torch.as_tensor(mask_np, device=device, dtype=torch.bool).unsqueeze(0)  # (1,N,K)

            action_b, logprob_b, value_b = model.act(obs.unsqueeze(0), mask)
            action = action_b.squeeze(0)
            logprob = logprob_b.squeeze(0)
            value = value_b.squeeze(0)
            
            # Since admission logic is removed, we just mock these values for logging
            adm_ratio_dl = 1.0
            adm_ratio_cf = 1.0
            adm_ratio_bk = 1.0

            next_obs_np, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            current_episode_reward += reward
            done = float(terminated or truncated)

            # Accumulate metrics
            total_adm_ratio_dl += adm_ratio_dl
            total_adm_ratio_cf += adm_ratio_cf
            total_adm_ratio_bk += adm_ratio_bk
            if info:
                total_mlu += info.get('mlu', 0.0)
                total_deadline_penalty += info.get('deadline_penalty', 0.0)
                total_coflow_cct += info.get('avg_coflow_cct', 0.0)
                total_change_cost += info.get('change_cost', 0.0)
                total_infeas_mass += info.get('infeas_mass', 0.0)
                total_capacity_scale += info.get('capacity_scale', 0.0)
                total_reward_abs += info.get('reward_abs', 0.0)
                total_reward_delta += info.get('reward_delta', 0.0)
                total_reward_mlu_cost += info.get('reward_mlu_cost', 0.0)
                total_reward_dl_miss_rate += info.get('reward_dl_miss_rate', 0.0)
                total_reward_delay_penalty += info.get('reward_delay_penalty', 0.0)
                total_reward_throughput_eff += info.get('reward_throughput_eff', 0.0)
                num_steps += 1
                
                # Accumulate episode metrics
                episode_deadline_miss += info.get('deadline_miss', 0.0)
                episode_deadline_total += info.get('deadline_total', 0.0)
                episode_coflow_cct_sum += info.get('avg_coflow_cct', 0.0) * info.get('num_coflows_done', 0.0)
                episode_coflow_count += info.get('num_coflows_done', 0.0)

            buffer.add(
                obs=obs,
                action=action,
                mask=mask.squeeze(0),
                logprob=logprob,
                reward=torch.tensor(reward, device=device, dtype=torch.float32),
                done=torch.tensor(done, device=device, dtype=torch.float32),
                value=value,
            )
            last_info = info
            obs = to_torch(next_obs_np, device)

            if done > 0.0:
                # Record episode reward and metrics
                episode_rewards.append(current_episode_reward)
                if writer:
                    writer.add_scalar('Episode/Reward', current_episode_reward, global_step)
                logging.info(f"Episode reward: {current_episode_reward:.4f} at step {global_step}")
                
                # Calculate and log episode metrics
                if episode_deadline_total > 0:
                    deadline_miss_rate = episode_deadline_miss / episode_deadline_total
                    if writer:
                        writer.add_scalar('Episode/Deadline_Miss_Rate', deadline_miss_rate, global_step)
                    logging.info(f"Episode deadline miss rate: {deadline_miss_rate:.3f}")
                
                if episode_coflow_count > 0:
                    avg_coflow_cct = episode_coflow_cct_sum / episode_coflow_count
                    if writer:
                        writer.add_scalar('Episode/Avg_Coflow_CCT', avg_coflow_cct, global_step)
                    logging.info(f"Episode average coflow CCT: {avg_coflow_cct:.3f}")
                
                # Reset episode metrics
                current_episode_reward = 0.0
                episode_deadline_miss = 0
                episode_deadline_total = 0
                episode_coflow_cct_sum = 0
                episode_coflow_count = 0
                
                obs_np, _ = env.reset()
                obs = to_torch(obs_np, device)

        with torch.no_grad():
            _mu, _ls, last_value = model.forward(obs.unsqueeze(0))
            last_value = last_value.squeeze(0)

        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)
        entropy_coef_now = cfg.entropy_coef * frac
        policy_loss, value_loss, entropy_loss, total_loss = ppo_update(
            model,
            optimizer,
            buffer,
            cfg,
            entropy_coef=entropy_coef_now,
        )

        if update % cfg.log_every_updates == 0:
            # Calculate average metrics
            avg_mlu = total_mlu / num_steps if num_steps > 0 else 0.0
            avg_deadline_penalty = total_deadline_penalty / num_steps if num_steps > 0 else 0.0
            avg_coflow_cct = total_coflow_cct / num_steps if num_steps > 0 else 0.0
            avg_change_cost = total_change_cost / num_steps if num_steps > 0 else 0.0
            avg_infeas_mass = total_infeas_mass / num_steps if num_steps > 0 else 0.0
            avg_capacity_scale = total_capacity_scale / num_steps if num_steps > 0 else 0.0
            avg_adm_ratio_dl = total_adm_ratio_dl / num_steps if num_steps > 0 else 0.0
            avg_adm_ratio_cf = total_adm_ratio_cf / num_steps if num_steps > 0 else 0.0
            avg_adm_ratio_bk = total_adm_ratio_bk / num_steps if num_steps > 0 else 0.0
            avg_reward_abs = total_reward_abs / num_steps if num_steps > 0 else 0.0
            avg_reward_delta = total_reward_delta / num_steps if num_steps > 0 else 0.0
            avg_reward_mlu_cost = total_reward_mlu_cost / num_steps if num_steps > 0 else 0.0
            avg_reward_dl_miss_rate = total_reward_dl_miss_rate / num_steps if num_steps > 0 else 0.0
            avg_reward_delay_penalty = total_reward_delay_penalty / num_steps if num_steps > 0 else 0.0
            avg_reward_throughput_eff = total_reward_throughput_eff / num_steps if num_steps > 0 else 0.0
            
            # Get mask average from last info if available
            avg_feasible = 0.0
            if last_info and "mask" in last_info:
                m = last_info["mask"]
                avg_feasible = m.sum(axis=1).mean()

            logging.info(
                f"update={update:4d} steps={global_step:7d} "
                f"mlu={avg_mlu:.3f} "
                f"deadline_pen={avg_deadline_penalty:.3f} "
                f"coflow_cct={avg_coflow_cct:.3f} "
                f"adm_dl={avg_adm_ratio_dl:.3f} "
                f"adm_cf={avg_adm_ratio_cf:.3f} "
                f"adm_bk={avg_adm_ratio_bk:.3f} "
                f"avg_feasible={avg_feasible:.2f} "
                f"r_abs={avg_reward_abs:.3f} "
                f"r_delta={avg_reward_delta:.3f} "
                f"r_mlu={avg_reward_mlu_cost:.3f} "
                f"r_dl={avg_reward_dl_miss_rate:.3f} "
                f"r_delay={avg_reward_delay_penalty:.3f} "
                f"r_thr={avg_reward_throughput_eff:.3f} "
                f"p_loss={policy_loss:.3f} "
                f"v_loss={value_loss:.3f} "
                f"ent={entropy_loss:.3f} "
                f"ent_coef={entropy_coef_now:.6f}"
            )
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Metrics/MLU', avg_mlu, global_step)
                writer.add_scalar('Metrics/Deadline_Penalty', avg_deadline_penalty, global_step)
                writer.add_scalar('Metrics/Coflow_CCT', avg_coflow_cct, global_step)
                writer.add_scalar('Metrics/Change_Cost', avg_change_cost, global_step)
                writer.add_scalar('Metrics/Infeas_Mass', avg_infeas_mass, global_step)
                writer.add_scalar('Metrics/Capacity_Scale', avg_capacity_scale, global_step)
                writer.add_scalar('Metrics/Avg_Feasible', avg_feasible, global_step)
            
            # Log losses
            if writer:
                writer.add_scalar('Losses/Policy_Loss', policy_loss, global_step)
                writer.add_scalar('Losses/Value_Loss', value_loss, global_step)
                writer.add_scalar('Losses/Entropy_Loss', entropy_loss, global_step)
                writer.add_scalar('Losses/Total_Loss', total_loss, global_step)

    # Log final episode reward if not yet recorded
    if current_episode_reward > 0:
        episode_rewards.append(current_episode_reward)
        if writer:
            writer.add_scalar('Episode/Reward', current_episode_reward, global_step)
        logging.info(f"Episode reward: {current_episode_reward:.4f} at step {global_step}")

    # Close TensorBoard writer
    if writer:
        writer.close()

    import os
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "ppo_te_agent_latest.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved successfully to {model_path}")
    print(f"Model saved successfully to {model_path}")

    logging.info("Training done.")
    return model, env


if __name__ == "__main__":
    print("Starting training...")
    print("Creating configuration...")
    
    # Enable debug mode for testing
    debug_mode = False
    
    if debug_mode:
        # Use smaller values for debugging
        cfg = PPOConfig(
            seed=0,
            device="cpu",  # set "cuda" if available
            total_steps=5000,
            rollout_steps=128,
            minibatch_size=64,
            update_epochs=4,
            n_demands=120,  # Cover all pairs
            k_paths=6,
        )
    else:
        # Use PPOConfig defaults to keep hyperparameters consistent with code updates.
        cfg = PPOConfig(
            seed=0,
            device="cpu",  # set "cuda" if available
            total_steps=1000_000,
        )
    
    print("Configuration created:")
    print(f"  Seed: {cfg.seed}")
    print(f"  Device: {cfg.device}")
    print(f"  Total steps: {cfg.total_steps}")
    print(f"  Rollout steps: {cfg.rollout_steps}")
    print(f"  Minibatch size: {cfg.minibatch_size}")
    print(f"  Update epochs: {cfg.update_epochs}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Number of demands: {cfg.n_demands}")
    print(f"  Number of paths: {cfg.k_paths}")
    
    print("Running train function...")
    model, env = train(cfg)
    print("Training completed successfully!")
