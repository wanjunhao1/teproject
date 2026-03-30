# env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from traffic import Flow, Coflow, TrafficGenerator, RealTrafficGenerator
from metrics import compute_mlu, compute_deadline_penalty, compute_coflow_cct
from common import Path

class RealTEEnv:
    """
    Real TE env skeleton (fixed-dim PPO-friendly):
      - fixed N demands per step (top-N (src,dst) aggregated by remaining bytes)
      - K candidate paths per (src,dst)
      - action: (N,K) simplex splits (flattened to N*K when passing to PPO)
      - mask: (N,K) bool feasible (e.g., deadline E2E delay budget feasible)
      - capacity enforcement: global scaling (simple hard-feasible baseline)

    IMPORTANT: this is a working skeleton. Replace:
      - _build_candidate_paths() with your K-shortest/ECMP/SR candidates
      - _compute_feasible_mask() with your deadline budget logic
      - _allocate_total_rates() with your slicing/allocation
      - _advance... with your precise per-flow/service model
    """

    def __init__(
        self,
        num_nodes: int,
        links: List[Tuple[int, int, float, float]],  # (u, v, capacity, prop_delay)
        candidate_paths: Dict[Tuple[int, int], List[Path]],
        n_demands: int,
        k_paths: int,
        episode_len: int,
        seed: int = 0,
        slot_duration_s: float = 1.0,
        w_deadline: float = 0.01,
        w_coflow: float = 0.02,
        w_mlu: float = 0.5,
        w_change: float = 0.1,
        w_infeas: float = 0.2,
        enforce_capacity: str = "scale",  # "scale" (simple), can extend
        use_real_traffic: bool = False,
        traffic_trace: str = "facebook_hadoop",
        # Dynamic features
        dynamic_capacity: bool = False,  # Enable time-varying link capacity
        capacity_variation: float = 0.1,  # Capacity variation ratio (±10%)
        link_failure_prob: float = 0.0,  # Probability of link failure
        traffic_burstiness: float = 0.0,  # Traffic burstiness factor
    ):
        self.num_nodes = num_nodes
        self.links = links
        self.paths = candidate_paths
        self.N = n_demands
        self.K = k_paths
        self.episode_len = episode_len
        self.slot_duration_s = slot_duration_s

        self.w_deadline = w_deadline
        self.w_coflow = w_coflow
        self.w_mlu = w_mlu
        self.w_change = 0.0  # 关闭或大幅降低路由跳动惩罚
        self.w_infeas = w_infeas

        self.enforce_capacity = enforce_capacity

        # Dynamic features
        self.dynamic_capacity = dynamic_capacity
        self.capacity_variation = capacity_variation
        self.link_failure_prob = link_failure_prob
        self.traffic_burstiness = traffic_burstiness

        self.rng = np.random.default_rng(seed)
        if use_real_traffic:
            self.traffic_gen = RealTrafficGenerator(
                self.rng, num_nodes, traffic_trace,
                burstiness=traffic_burstiness
            )
        else:
            self.traffic_gen = TrafficGenerator(self.rng, num_nodes)

        self.num_links = len(links)
        self.base_link_cap = np.array([c for (_, _, c, _) in links], dtype=np.float32)
        self.link_cap = self.base_link_cap.copy()  # Current capacity (may vary)
        self.link_delay = np.array([d for (_, _, _, d) in links], dtype=np.float32)
        self.link_failure_status = np.zeros(self.num_links, dtype=bool)  # Link failure status

        # sanity: ensure each (src,dst) has exactly K candidate paths
        for key, plist in self.paths.items():
            if len(plist) != self.K:
                raise ValueError(f"candidate_paths[{key}] has {len(plist)} != K={self.K}")

        # dynamics
        self.t = 0
        self.link_load = np.zeros((self.num_links,), dtype=np.float32)

        # active traffic
        self.active_flows: List[Flow] = []
        self.active_coflows: Dict[int, Coflow] = {}

        # last action for change cost
        self.prev_action: Optional[np.ndarray] = None

        # current demands snapshot (for mask)
        self.current_demands: List[Tuple[int, int, float, List[Flow]]] = []
        self.current_mask = np.ones((self.N, self.K), dtype=np.bool_)
        self.prev_mlu: float = 0.0
        self.prev_dl_miss_rate: float = 0.0

    # ---- shapes ----
    @property
    def num_demands_per_step(self) -> int:
        return self.N

    @property
    def k_paths(self) -> int:
        return self.K

    @property
    def action_dim(self) -> int:
        return self.N * self.K

    @property
    def obs_dim(self) -> int:
        # util(L) + summary(12) + per-demand summary(3*N)
        return self.num_links + 12 + 3 * self.N

    # ---- external API ----
    def get_mask(self) -> np.ndarray:
        return self.current_mask.copy()

    def reset(self) -> Tuple[np.ndarray, Dict]:
        self.t = 0
        self.link_load.fill(0.0)
        self.active_flows.clear()
        self.active_coflows.clear()
        self.prev_action = None
        self.current_demands = []
        self.current_mask = np.ones((self.N, self.K), dtype=np.bool_)
        self.prev_mlu = 0.0
        self.prev_dl_miss_rate = 0.0

        # warm start - ingest traffic for t=0
        self._ingest_new_traffic()
        self._refresh_demands_and_mask()
        return self._get_obs(), {}

    def step(self, action_flat: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._update_link_capacity()
        # Action is entirely routing now
        routing_action = action_flat[:self.N * self.K]
        
        # Calculate admission ratios using sigmoid (Phase 1: Force to 1.0 to freeze admission)
        adm_ratio_dl = 1.0
        adm_ratio_cf = 1.0
        adm_ratio_bk = 1.0
        A = routing_action.reshape(self.N, self.K).astype(np.float32)
        A = np.clip(A, 1e-8, 1.0)
        A = A / A.sum(axis=1, keepdims=True)

        # penalty if policy put mass on infeasible paths (before env fixes)
        infeas_mass = float(np.sum(A * (~self.current_mask)))

        # env enforces mask: zero infeasible + renorm
        A = A * self.current_mask.astype(np.float32)
        row_sum = A.sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-8] = 1.0
        A = A / row_sum

        # change cost
        change_cost = 0.0
        if self.prev_action is not None:
            change_cost = float(np.sum(np.abs(A - self.prev_action)))
        self.prev_action = A.copy()

        # choose total rates per demand (bytes per slot here)
        demand_rates = self._allocate_total_rates(self.current_demands)  # (N,)

        # per-path bytes this slot
        x = A * demand_rates[:, None]  # (N,K)
        requested_total = float(np.sum(demand_rates)) + 1e-9

        # compute link load
        self.link_load.fill(0.0)
        self._accumulate_link_load(self.current_demands, x)

        # enforce capacity (per-link local scaling so max util<=1 without choking global network)
        if self.enforce_capacity == "scale":
            util = self.link_load / (self.link_cap + 1e-9)
            if np.max(util) > 1.0:
                for i, (s, d, _rem, _flows) in enumerate(self.current_demands):
                    if (s, d) not in self.paths: continue
                    for k, path in enumerate(self.paths[(s, d)]):
                        max_u = max([util[e] for e in path.links] + [1.0])
                        if max_u > 1.0:
                            x[i, k] /= max_u
                
                # Recompute exact link loads locally scaled
                self.link_load.fill(0.0)
                self._accumulate_link_load(self.current_demands, x)

            achieved_total = float(np.sum(x))
            throughput_eff = float(np.clip(achieved_total / requested_total, 0.0, 1.0))

        # Calculate real-time deadline penalty for active flows
        real_time_deadline_miss = 0
        real_time_deadline_total = 0
        flows_to_drop = []
        max_tolerance = 4  # 降低容忍极限，逼迫SPF在严重延迟时触发毁毁性扣分（原为10）

        for f in self.active_flows:
            if f.cls == "deadline" and f.deadline_t is not None:
                if self.t > f.deadline_t:
                    # Calculate real-time penalty for each slot of delay
                    lateness = self.t - f.deadline_t
                    # Check if flow should be dropped due to excessive lateness
                    if lateness > max_tolerance:
                        flows_to_drop.append(f)

        # Drop flows that have exceeded maximum tolerance
        for f in flows_to_drop:
            # Add huge penalty for dropped flows
            real_time_deadline_miss += 1
            real_time_deadline_total += 1
        self.active_flows = [f for f in self.active_flows if f not in flows_to_drop]

        # advance traffic, collect done
        done_flows, done_coflows = self._advance_and_collect_done(self.current_demands, x)

        # metrics
        mlu = compute_mlu(self.link_load, self.link_cap)
        dl_stats = compute_deadline_penalty(done_flows, self.active_flows, finished_t=self.t)
        # We manually append drops which are completely wiped
        dl_stats["deadline_penalty"] += len(flows_to_drop) * 10.0
        dl_stats["deadline_miss"] += real_time_deadline_miss
        dl_stats["deadline_total"] += real_time_deadline_total
        cf_stats = compute_coflow_cct(done_coflows)

        # Print detailed metrics to console

        # Check deadline flows that are about to miss
        deadline_flows = [f for f in self.active_flows if f.cls == "deadline" and f.deadline_t is not None]
        about_to_miss = [f for f in deadline_flows if f.deadline_t <= self.t + 1]
        already_missed = [f for f in deadline_flows if f.deadline_t < self.t]

        # Check coflows that are almost done
        almost_done_coflows = [cf for cfid, cf in self.active_coflows.items() if sum(1 for f in cf.flows if f.remaining_bytes > 0) <= 2]

        # time forward + arrivals
        self.t += 1  # Increment time first
        new_flows, new_coflows = self.traffic_gen.sample_slot(self.t)
        
        # Admission Control: Let the agent decide based on network state
        max_active_flows = 300  # Maximum number of active flows
        
        # Calculate how many new flows we can accept (no hard limit on new flows per slot)
        current_active = len(self.active_flows)
        available_slots = max(0, max_active_flows - current_active)
        
        deadline_flows = [f for f in new_flows if f.cls == "deadline"]
        coflow_flows = [f for f in new_flows if f.cls == "coflow"]
        bulk_flows = [f for f in new_flows if f.cls == "bulk"]

        # Apply the agent's class-based admission ratio decisions
        accept_target_dl = int(len(deadline_flows) * adm_ratio_dl)
        accept_target_cf = int(len(coflow_flows) * adm_ratio_cf)
        accept_target_bk = int(len(bulk_flows) * adm_ratio_bk)
        


        # Accept flows in priority order
        accepted_flows = []
        
        # Accept deadline flows first
        actual_accept_dl = min(accept_target_dl, available_slots)
        accepted_flows.extend(deadline_flows[:actual_accept_dl])
        available_slots -= actual_accept_dl
        
        # Accept coflow flows next
        actual_accept_cf = min(accept_target_cf, available_slots)
        if actual_accept_cf > 0:
            accepted_flows.extend(coflow_flows[:actual_accept_cf])
            available_slots -= actual_accept_cf
        
        # Accept bulk flows last
        actual_accept_bk = min(accept_target_bk, available_slots)
        if actual_accept_bk > 0:
            accepted_flows.extend(bulk_flows[:actual_accept_bk])
            available_slots -= actual_accept_bk
        
        # Accept all new coflows (since they contain multiple flows)
        # But limit coflow acceptance if too many active flows
        accepted_coflows = []
        max_active_coflows = 50  # Maximum number of active coflows
        current_active_coflows = len(self.active_coflows)
        coflow_slots = max(0, max_active_coflows - current_active_coflows)
        accepted_coflows.extend(new_coflows[:min(len(new_coflows), coflow_slots)])
        
        # Add accepted flows and coflows
        self.active_flows.extend(accepted_flows)
        for cf in accepted_coflows:
            self.active_coflows[cf.coflow_id] = cf
        
        # Calculate class-specific rejections
        rejected_dl = len(deadline_flows) - actual_accept_dl
        rejected_cf = len(coflow_flows) - actual_accept_cf
        rejected_bk = len(bulk_flows) - actual_accept_bk
        rejected_coflows = len(new_coflows) - len(accepted_coflows)
        
        # Log admission control decisions

        # Add rejected flows to tracking before calculating misses
        dl_stats["deadline_total"] += rejected_dl
        dl_stats["deadline_miss"] += rejected_dl

        # 1. MLU: relaxed safe zone, steeper penalty beyond 0.8
        if mlu <= 0.8:
            mlu_cost = mlu * 0.25 # up to 0.2
        else:
            mlu_cost = 0.2 + (mlu - 0.8) * 2.0
        mlu_cost = min(mlu_cost, 1.5)  # clip

        # 2. Deadline miss rate: 每步的
        active_dl = [f for f in self.active_flows if f.cls == "deadline" and f.deadline_t is not None]
        n_active_dl = len(active_dl)
        n_missed_dl = sum(1 for f in active_dl if self.t > f.deadline_t)
        dl_miss_rate = n_missed_dl / max(n_active_dl, 1)  # [0, 1]

        # 3. Throughput completion ratio (secondary shaping)
        n_completed = len(done_flows)
        n_active = max(len(self.active_flows), 1)
        completion_rate = min(n_completed / n_active, 1.0)  # [0, 1]

        # 4. Routing-delay penalty for deadline demands.
        # Encourages assigning deadline traffic to lower-latency paths.
        delay_penalty = 0.0
        deadline_groups = 0
        for i, (s, d, _rem, flows) in enumerate(self.current_demands):
            if not flows or (s, d) not in self.paths:
                continue
            if not any(f.cls == "deadline" for f in flows):
                continue
            delays = np.array([p.prop_delay for p in self.paths[(s, d)]], dtype=np.float32)
            min_delay = float(np.min(delays))
            expected_delay = float(np.sum(A[i] * delays))
            norm = max(min_delay, 1e-6)
            delay_penalty += max(0.0, (expected_delay - min_delay) / norm)
            deadline_groups += 1
        if deadline_groups > 0:
            delay_penalty /= deadline_groups

        # Action-sensitive reward: prioritize controllable routing quality over near-constant throughput.
        reward_abs = (
            0.20 * throughput_eff
            - 0.55 * mlu_cost
            - 0.20 * dl_miss_rate
            - 0.10 * delay_penalty
            + 0.05 * completion_rate
        )
        reward_delta = 0.10 * (self.prev_mlu - mlu) + 0.10 * (self.prev_dl_miss_rate - dl_miss_rate)
        reward = reward_abs + reward_delta - 0.01 * (change_cost / max(self.N, 1)) - 0.02 * infeas_mass
        reward = float(np.clip(reward, -2.0, 2.0))

        self.prev_mlu = float(mlu)
        self.prev_dl_miss_rate = float(dl_miss_rate)

        terminated = False
        truncated = self.t >= self.episode_len

        # REFRESH DEMANDS FOR NEXT STEP AND OBSERVATION
        self._refresh_demands_and_mask()

        info: Dict[str, float | np.ndarray] = {
            "t": float(self.t),
            "mlu": float(mlu),
            "deadline_penalty": float(dl_stats["deadline_penalty"]),
            "deadline_miss": float(dl_stats["deadline_miss"]),
            "deadline_total": float(dl_stats["deadline_total"]),
            "avg_coflow_cct": float(cf_stats["avg_coflow_cct"]),
            "num_coflows_done": float(cf_stats["num_coflows_done"]),
            "change_cost": float(change_cost),
            "infeas_mass": float(infeas_mass),
            "capacity_scale": 1.0,
            "mask": self.current_mask.copy(),
            "adm_ratio_dl": adm_ratio_dl,
            "adm_ratio_cf": adm_ratio_cf,
            "adm_ratio_bk": adm_ratio_bk,
            "reward_abs": float(reward_abs),
            "reward_delta": float(reward_delta),
            "reward_mlu_cost": float(mlu_cost),
            "reward_dl_miss_rate": float(dl_miss_rate),
            "reward_completion_rate": float(completion_rate),
            "reward_throughput_eff": float(throughput_eff),
            "reward_delay_penalty": float(delay_penalty),
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ---- internals ----
    def _ingest_new_traffic(self):
        new_flows, new_coflows = self.traffic_gen.sample_slot(self.t)
        
        # Admission Control: Let the agent decide based on network state
        max_active_flows = 300  # Maximum number of active flows
        
        # Calculate how many new flows we can accept (no hard limit on new flows per slot)
        current_active = len(self.active_flows)
        available_slots = max(0, max_active_flows - current_active)
        flows_to_accept = min(len(new_flows), available_slots)
        
        # Prioritize accepting deadline flows first
        deadline_flows = [f for f in new_flows if f.cls == "deadline"]
        coflow_flows = [f for f in new_flows if f.cls == "coflow"]
        bulk_flows = [f for f in new_flows if f.cls == "bulk"]
        
        # Accept flows in priority order
        accepted_flows = []
        
        # Accept deadline flows first
        accepted_flows.extend(deadline_flows[:min(len(deadline_flows), flows_to_accept)])
        remaining_slots = flows_to_accept - len(accepted_flows)
        
        if remaining_slots > 0:
            # Accept coflow flows next
            accepted_flows.extend(coflow_flows[:min(len(coflow_flows), remaining_slots)])
            remaining_slots = flows_to_accept - len(accepted_flows)
        
        if remaining_slots > 0:
            # Accept bulk flows last
            accepted_flows.extend(bulk_flows[:min(len(bulk_flows), remaining_slots)])
        
        # Accept all new coflows (since they contain multiple flows)
        # But limit coflow acceptance if too many active flows
        accepted_coflows = []
        max_active_coflows = 50  # Maximum number of active coflows
        current_active_coflows = len(self.active_coflows)
        coflow_slots = max(0, max_active_coflows - current_active_coflows)
        accepted_coflows.extend(new_coflows[:min(len(new_coflows), coflow_slots)])
        
        # Add accepted flows and coflows
        self.active_flows.extend(accepted_flows)
        for cf in accepted_coflows:
            self.active_coflows[cf.coflow_id] = cf
        
        # Log admission control decisions

    def _update_link_capacity(self):
        """Update link capacity dynamically based on time and failure status"""
        if not self.dynamic_capacity and self.link_failure_prob == 0:
            return
        
        # Time-varying capacity (sinusoidal pattern + noise)
        if self.dynamic_capacity:
            time_factor = 1.0 + self.capacity_variation * np.sin(2 * np.pi * self.t / 100)
            noise = self.rng.uniform(-self.capacity_variation * 0.5, self.capacity_variation * 0.5, self.num_links)
            self.link_cap = self.base_link_cap * (time_factor + noise)
            self.link_cap = np.maximum(self.link_cap, self.base_link_cap * 0.5)  # Min 50% capacity
        
        # Link failures
        if self.link_failure_prob > 0:
            for i in range(self.num_links):
                if self.rng.random() < self.link_failure_prob:
                    self.link_failure_status[i] = not self.link_failure_status[i]
            # Set failed links to 0 capacity
            self.link_cap = np.where(self.link_failure_status, 0, self.link_cap)

    def _refresh_demands_and_mask(self):
        self.current_demands = self._select_demands(self.N)
        self.current_mask = self._compute_feasible_mask(self.current_demands)

    def _select_demands(self, N: int) -> List[Tuple[int, int, float, List[Flow]]]:
        # aggregate by (src,dst) with remaining sum
        agg: Dict[Tuple[int, int], float] = {}
        meta: Dict[Tuple[int, int], List[Flow]] = {}
        has_deadline: Dict[Tuple[int, int], bool] = {}
        has_coflow: Dict[Tuple[int, int], bool] = {}
        deadline_count: Dict[Tuple[int, int], int] = {}
        coflow_count: Dict[Tuple[int, int], int] = {}
        urgency_score: Dict[Tuple[int, int], float] = {}
        is_small_flow: Dict[Tuple[int, int], bool] = {}
        earliest_deadline: Dict[Tuple[int, int], int] = {}
        
        for f in self.active_flows:
            if f.remaining_bytes <= 1.0:
                continue
            key = (f.src, f.dst)
            agg[key] = agg.get(key, 0.0) + float(f.remaining_bytes)
            meta.setdefault(key, []).append(f)
            if f.cls == "deadline":
                has_deadline[key] = True
                deadline_count[key] = deadline_count.get(key, 0) + 1
                # Calculate urgency for deadline flows
                if f.deadline_t is not None:
                    time_left = f.deadline_t - self.t
                    if time_left > 0:
                        urgency = max(1.0, 1000.0 / time_left)  # Increased urgency factor
                        urgency_score[key] = max(urgency_score.get(key, 0.0), urgency)
                        # Track earliest deadline for EDF
                        if key not in earliest_deadline or f.deadline_t < earliest_deadline[key]:
                            earliest_deadline[key] = f.deadline_t
                    else:
                        # Already past deadline, highest urgency
                        urgency_score[key] = 10000.0  # Increased penalty
                        earliest_deadline[key] = self.t - 1  # Ensure highest priority
            if f.cls == "coflow":
                has_coflow[key] = True
                coflow_count[key] = coflow_count.get(key, 0) + 1
            # Check if this is a small flow (almost done)
            if f.remaining_bytes < 10e6:  # Less than 10MB
                is_small_flow[key] = True

        # Keep demand selection simple to avoid hard-coded scheduling dominating PPO learning.
        def sort_key(k):
            priority = agg[k]

            if has_deadline.get(k, False):
                priority *= 1.30
                if k in earliest_deadline:
                    time_to_deadline = earliest_deadline[k] - self.t
                    priority += max(0.0, 20.0 - float(time_to_deadline)) * 1e6
            elif has_coflow.get(k, False):
                priority *= 1.15

            if is_small_flow.get(k, False):
                priority *= 1.10

            return priority

        keys = sorted(agg.keys(), key=sort_key, reverse=True)[:N]
        demands: List[Tuple[int, int, float, List[Flow]]] = []
        for k in keys:
            demands.append((k[0], k[1], agg[k], meta[k]))

        # pad to fixed N
        while len(demands) < N:
            demands.append((0, 1, 0.0, []))
        
        # Print debug info
        
        return demands

    def _compute_feasible_mask(self, demands: List[Tuple[int, int, float, List[Flow]]]) -> np.ndarray:
        """
        Feasibility rule with priority for low-latency paths:
          - If demand contains at least one deadline flow, compute a delay budget = min(deadline_t - now_t) in slots
          - A path is feasible if its prop_delay <= budget (in seconds)
          - For all demands, prioritize low-latency paths
        """
        mask = np.ones((self.N, self.K), dtype=np.bool_)
        now_t = self.t

        for i, (s, d, _rem, flows) in enumerate(demands):
            if not flows or (s, d) not in self.paths:
                # Padded/inactive demand: make it deterministic to avoid injecting policy noise.
                mask[i, :] = False
                mask[i, 0] = True
                continue

            plist = self.paths[(s, d)]
            path_delays = [p.prop_delay for p in plist]
            
            # find if any deadline flow inside this aggregated demand
            dl_deadlines = [f.deadline_t for f in flows if f.cls == "deadline" and f.deadline_t is not None]
            if dl_deadlines:
                # remaining budget in slots
                budget_slots = max(0, min(dl_deadlines) - now_t)
                # map slots -> seconds; extremely simplified
                budget_s = float(budget_slots) * self.slot_duration_s

                feas = []
                for p in plist:
                    feas.append(p.prop_delay <= budget_s + 1e-9)
                feas = np.array(feas, dtype=np.bool_)

                # For deadline demands, keep only low-latency candidates among feasible ones.
                # This reduces near-equivalent high-delay choices and strengthens learning signal.
                top_k = min(3, len(plist))
                low_latency_idx = np.argsort(path_delays)[:top_k]
                latency_pref = np.zeros(len(plist), dtype=np.bool_)
                latency_pref[low_latency_idx] = True
                feas = feas & latency_pref
            else:
                # For non-deadline flows, all paths are feasible, but we'll prioritize low-latency ones
                feas = np.ones(len(plist), dtype=np.bool_)

            # ensure at least 1 feasible to avoid dead action
            if feas.sum() == 0:
                # Select the path with minimum delay
                min_delay_idx = np.argmin(path_delays)
                feas[min_delay_idx] = True
            # encourage >=2 feasible if possible (helps exploration), but prioritize low-latency
            if feas.sum() == 1 and self.K > 1:
                # Find the second lowest delay path
                sorted_indices = np.argsort(path_delays)
                for idx in sorted_indices:
                    if not feas[idx]:
                        feas[idx] = True
                        break

            mask[i, :] = feas

        return mask

    def _allocate_total_rates(self, demands: List[Tuple[int, int, float, List[Flow]]]) -> np.ndarray:
        """
        Total bytes served per demand per slot.
        Replace with your slicing model across classes / fairness / max-min.
        """
        # Adjusted global budget to match actual dynamically loaded link capacity
        global_budget = (float(np.sum(self.link_cap)) / 8.0) * 0.95  # bps -> bytes/slot
        N = len(demands)
        rates = np.zeros((N,), dtype=np.float32)
        if N == 0:
            return rates

        # Calculate priority-based shares with higher weights
        priorities = []
        for i, (_s, _d, rem, flows) in enumerate(demands):
            priority = 1.0  # Default priority for bulk flows
            deadline_count = sum(1 for f in flows if f.cls == "deadline")
            coflow_count = sum(1 for f in flows if f.cls == "coflow")
            
            # Check for "last tail" flows
            is_last_tail = False
            for f in flows:
                # Check if this is the last flow in a coflow
                if f.coflow_id is not None:
                    coflow = None
                    for cfid, cf in self.active_coflows.items():
                        if cfid == f.coflow_id:
                            coflow = cf
                            break
                    if coflow:
                        active_flows_in_coflow = sum(1 for cf_flow in coflow.flows if cf_flow.remaining_bytes > 0)
                        if active_flows_in_coflow <= 2:
                            is_last_tail = True
                # Check if flow is almost done
                if f.remaining_bytes < 1e6:  # Less than 1MB remaining
                    is_last_tail = True
            
            if deadline_count > 0:
                # Keep deadline preference, but avoid overpowering all other routing decisions.
                priority = 2.5 + 0.3 * deadline_count
                # Add bounded urgency factor based on how close the deadline is.
                for f in flows:
                    if f.cls == "deadline" and f.deadline_t is not None:
                        time_left = f.deadline_t - self.t
                        if time_left > 0:
                            urgency = np.clip(4.0 / max(time_left, 1), 1.0, 2.5)
                            priority *= urgency
                        else:
                            priority *= 2.5
            elif coflow_count > 0:
                # Moderate preference for coflow progress.
                priority = 1.8 + 0.2 * coflow_count
                if is_last_tail:
                    priority *= 1.2
            else:
                # Bulk flows
                priority = 1.0
                if is_last_tail:
                    priority *= 1.1
            
            # Apply last tail bonus
            if is_last_tail:
                priority *= 1.15
                
            priorities.append(priority)
        
# Waterfilling allocation to prevent bandwidth waste
        remaining_budget = global_budget
        unfulfilled = list(range(N))
        
        while unfulfilled and remaining_budget > 1e6:
            current_total_prio = sum(priorities[i] for i in unfulfilled)
            if current_total_prio == 0:
                current_total_prio = len(unfulfilled)
                
            budget_this_round = remaining_budget
            next_unfulfilled = []
            
            for i in unfulfilled:
                rem = demands[i][2]
                share = budget_this_round * (priorities[i] / current_total_prio)
                # Ensure minimum bandwidth for all demands
                share = max(share, budget_this_round * 0.001)
                
                needed = rem - rates[i]
                allocation = min(share, needed)
                # Cap allocation just in case
                allocation = min(allocation, remaining_budget)
                
                rates[i] += allocation
                remaining_budget -= allocation
                
                # If flow still needs more and didn't get its full rem
                if rates[i] < rem - 1.0 and share < needed:
                    next_unfulfilled.append(i)
                    
            if len(next_unfulfilled) == len(unfulfilled):
                # No one finished cleanly this round, break to prevent infinite loop
                break
                
            unfulfilled = next_unfulfilled

        return rates.astype(np.float32)

    def _accumulate_link_load(self, demands, x: np.ndarray):
        # x: (N,K) bytes per slot, add to link load
        for i, (s, d, _rem, _flows) in enumerate(demands):
            if (s, d) not in self.paths:
                continue
            plist = self.paths[(s, d)]
            for k, path in enumerate(plist):
                rate = float(x[i, k])
                if rate <= 0:
                    continue
                for e in path.links:
                    self.link_load[e] += rate

    def _capacity_scale_factor(self, eps: float = 1e-9) -> float:
        util = self.link_load / (self.link_cap + eps)
        m = float(np.max(util)) if util.size else 0.0
        if m <= 1.0:
            return 1.0
        return 1.0 / m

    def _advance_and_collect_done(self, demands, x: np.ndarray):
        """
        Simple drain model: For each aggregated (src,dst), distribute achieved bytes proportionally
        to constituent flows by remaining bytes.
        Replace with your precise scheduling model if needed.
        """
        done_flows: List[Flow] = []
        done_coflows: List[Coflow] = []

        achieved = x.sum(axis=1)  # bytes per demand in this slot

        # Process demands and track which flows have been processed
        processed_flow_ids = set()
        for i, (_s, _d, _rem, flows) in enumerate(demands):
            if not flows:
                continue
            bytes_i = float(achieved[i])
            if bytes_i <= 0:
                continue

            total_rem = sum(max(0.0, f.remaining_bytes) for f in flows)
            if total_rem <= 0:
                continue

            for f in flows:
                if f.remaining_bytes <= 1.0:  # Tolerate small floating point residuals
                    f.remaining_bytes = 0.0
                    done_flows.append(f)
                    processed_flow_ids.add(f.flow_id)
                    continue
                take = bytes_i * (f.remaining_bytes / total_rem)
                f.remaining_bytes = max(0.0, f.remaining_bytes - take)
                if f.remaining_bytes <= 1.0:  # Tolerate small floating point residuals
                    f.remaining_bytes = 0.0
                    done_flows.append(f)
                processed_flow_ids.add(f.flow_id)

        # Check all active flows to see if any have finished (including those not in demands)
        for f in self.active_flows:
            if f.flow_id not in processed_flow_ids and f.remaining_bytes <= 1.0:
                f.remaining_bytes = 0.0
                done_flows.append(f)

        # coflow done detection + set finished_t
        finished_cfids = []
        for cfid, cf in self.active_coflows.items():
            if cf.done():
                cf.finished_t = self.t
                done_coflows.append(cf)
                finished_cfids.append(cfid)
        for cfid in finished_cfids:
            self.active_coflows.pop(cfid, None)

        # remove finished flows
        self.active_flows = [f for f in self.active_flows if f.remaining_bytes > 1.0]
        
        # Print debug info
        
        return done_flows, done_coflows

    def _get_obs(self) -> np.ndarray:
        util = self.link_load / (self.link_cap + 1e-9)
        util = np.clip(util, 0.0, 3.0).astype(np.float32)

        # summary
        n_deadline = sum(1 for f in self.active_flows if f.cls == "deadline" and f.remaining_bytes > 0)
        n_coflow = len(self.active_coflows)
        n_bulk = sum(1 for f in self.active_flows if f.cls == "bulk" and f.remaining_bytes > 0)

        rem_deadline = sum(f.remaining_bytes for f in self.active_flows if f.cls == "deadline")
        rem_coflow = sum(f.remaining_bytes for f in self.active_flows if f.cls == "coflow")
        rem_bulk = sum(f.remaining_bytes for f in self.active_flows if f.cls == "bulk")

        # Calculate total demand rate and capacity ratio
        total_demand = sum(d[2] for d in self.current_demands)
        total_capacity = sum(self.link_cap)
        demand_capacity_ratio = total_demand / (5e9)  # 5Gbps as reference capacity
        demand_capacity_ratio = np.clip(demand_capacity_ratio, 0.0, 3.0)
        
        # Calculate queue length (number of active flows)
        queue_length = len(self.active_flows)
        queue_length_normalized = queue_length / 100.0  # Normalize to 0-3 range
        queue_length_normalized = np.clip(queue_length_normalized, 0.0, 3.0)

        # Time urgency features for active flows
        active_deadline_flows = [f for f in self.active_flows if f.cls == "deadline" and f.deadline_t is not None]
        min_time_to_deadline = 20.0  # default safe value
        danger_flows_count = 0.0
        
        if active_deadline_flows:
            # find minimum time to deadline
            times_to_dl = [f.deadline_t - self.t for f in active_deadline_flows]
            min_time_to_deadline = float(min(times_to_dl))
            # count flows that are in danger (<= 3 slots to deadline)
            danger_flows_count = float(sum(1 for t in times_to_dl if t <= 3))
            
        # normalize to be roughly in similar scales
        min_time_to_deadline = float(np.clip(min_time_to_deadline, -10.0, 20.0))

        agg = np.array(
            [
                float(self.t) / self.episode_len,                    # [0, 1]
                float(np.max(util) if util.size else 0.0),           # [0, 3] (已clip)
                float(np.mean(util) if util.size else 0.0),          # [0, 3]
                float(np.std(util) if util.size else 0.0),           # [0, ~1]
                float(n_deadline) / 50.0,                            # 归一化
                float(n_coflow) / 50.0,                              # 归一化
                float(n_bulk) / 100.0,                               # 归一化
                float(self.current_mask.sum(axis=1).mean()) / self.K, # [0, 1]
                float(demand_capacity_ratio),                        # 已clip [0, 3]
                # 删掉 total_demand 和 total_capacity 的裸值！
                float(queue_length_normalized),                      # 已归一化
                float(np.clip(min_time_to_deadline / 20.0, -0.5, 1.0)), # 归一化
                float(danger_flows_count) / 20.0,                    # 归一化
            ],
            dtype=np.float32,
        )

        # per-demand stats: remaining(MB), has_deadline(0/1), feasible_count
        per = []
        for (s, d, rem, flows) in self.current_demands:
            has_dl = 1.0 if any(f.cls == "deadline" for f in flows) else 0.0
            # log(1 + rem/1e6) keeps values within 0~5 range even for 5GB drops
            normalized_rem = np.log1p(float(rem / 1e6))
            per.extend([float(normalized_rem), has_dl, 0.0])  # feasible_count filled below
        per = np.array(per, dtype=np.float32).reshape(self.N, 3)
        per[:, 2] = self.current_mask.sum(axis=1).astype(np.float32)
        per = per.reshape(-1)

        return np.concatenate([util, agg, per], axis=0)



