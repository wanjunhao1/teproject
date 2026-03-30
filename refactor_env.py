import re

with open('env.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix _update_link_capacity 
content = content.replace(
    'def step(self, action_flat: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:\n',
    'def step(self, action_flat: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:\n        self._update_link_capacity()\n'
)

# 2. Fix print statements in step() and _select_demands()
# Just remove lines starting with spaces followed by `print(`
content = re.sub(r'^[ \t]*print\(.*?\)\n', '', content, flags=re.MULTILINE)

# 3. Rewrite _ingest_new_traffic to be simpler (just accept all for init)
old_ingest = '''    def _ingest_new_traffic(self):
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
        flows_to_accept -= len(accepted_flows)

        # Accept coflow flows next
        if flows_to_accept > 0:
            accepted_cf = coflow_flows[:min(len(coflow_flows), flows_to_accept)]
            accepted_flows.extend(accepted_cf)
            flows_to_accept -= len(accepted_cf)

        # Accept bulk flows last
        if flows_to_accept > 0:
            accepted_bk = bulk_flows[:min(len(bulk_flows), flows_to_accept)]
            accepted_flows.extend(accepted_bk)

        # Accept all new coflows
        accepted_coflows = new_coflows

        self.active_flows.extend(accepted_flows)
        for cf in accepted_coflows:
            self.active_coflows[cf.coflow_id] = cf'''

new_ingest = '''    def _ingest_new_traffic(self):
        # Simply accept all traffic initially (at t=0) to jumpstart the environment.
        new_flows, new_coflows = self.traffic_gen.sample_slot(self.t)
        self.active_flows.extend(new_flows)
        for cf in new_coflows:
            self.active_coflows[cf.coflow_id] = cf'''
content = content.replace(old_ingest, new_ingest)

# 4. Fix reward calculation
old_reward = '''        idle_penalty_scaler = 1.0
        if mlu < 0.4 and current_active < 50:
            idle_penalty_scaler = 10.0  # Force it to take flows if network is idle

# Refined class-based penalty:
        # Prevent the "Reject to avoid late penalty" loophole.
        # Rejecting Deadline is extremely bad (-500) because missing the deadline could cost up to -1000.
        # Rejecting Coflow is medium (-250).
        # Rejecting Bulk is very light (-5) to encourage dropping bulk when congested.
        admission_penalty = (
            (rejected_dl * 500.0) +
            (rejected_cf * 250.0) +
            (rejected_bk * 5.0) +
            (rejected_coflows * 250.0)
        ) * idle_penalty_scaler

        # Calculate drop penalty for flows dropped due to excessive lateness    
        drop_penalty = len(flows_to_drop) * 200.0  # Even heavier penalty for dropping

        # Add nonlinear congestion penalty
        congestion_penalty = 0.0
        if mlu > 0.8:
            # Exponential penalty for congestion
            congestion_penalty = 1000.0 * (mlu - 0.8) ** 2  # Nonlinear penalty that increases rapidly

        # Add reward for safe network utilization (encourage throughput)        
        utilization_reward = 0.0
        if mlu <= 0.8:
            # Reward higher utilization as long as it's safe (max +40 at mlu=0.8)
            utilization_reward = 50.0 * mlu

        reward = -(
            self.w_deadline * dl_stats["deadline_penalty"]
            + self.w_coflow * cf_stats["avg_coflow_cct"]
            + self.w_mlu * mlu
            + self.w_change * change_cost
            + self.w_infeas * infeas_mass
            + admission_penalty  # Dynamic penalty for rejection
            + drop_penalty  # Heavy penalty for dropping
            + congestion_penalty  # Penalty for congestion
        )

        # Add structured reward for completed flows based on class
        completed_dl = sum(1 for f in done_flows if f.cls == "deadline")        
        completed_cf = sum(1 for f in done_flows if f.cls == "coflow")
        completed_bk = sum(1 for f in done_flows if f.cls == "bulk")
        completed_reward = (completed_dl * 500.0) + (completed_cf * 200.0) + (completed_bk * 50.0)
        reward += completed_reward + utilization_reward  # Add utilization reward'''

new_reward = '''        idle_penalty_scaler = 1.0
        if mlu < 0.4 and current_active < 50:
            idle_penalty_scaler = 2.0  # Encourage taking flows if network is idle

        # Substantially normalized penalties
        admission_penalty = (
            (rejected_dl * 5.0) +
            (rejected_cf * 2.0) +
            (rejected_bk * 0.1) +
            (rejected_coflows * 2.0)
        ) * idle_penalty_scaler

        drop_penalty = len(flows_to_drop) * 10.0

        # Milder nonlinear congestion penalty
        congestion_penalty = 0.0
        if mlu > 0.8:
            congestion_penalty = 50.0 * (mlu - 0.8) ** 2

        # Reward safe network utilization
        utilization_reward = 0.0
        if mlu <= 0.8:
            utilization_reward = 2.0 * mlu

        # Re-scaled base rewards to prevent gradient domination
        reward = -(
            self.w_deadline * dl_stats["deadline_penalty"] * 0.1
            + self.w_coflow * cf_stats["avg_coflow_cct"] * 0.1
            + self.w_mlu * mlu * 5.0
            + self.w_change * change_cost
            + self.w_infeas * infeas_mass
            + admission_penalty
            + drop_penalty
            + congestion_penalty
        )

        completed_dl = sum(1 for f in done_flows if f.cls == "deadline")        
        completed_cf = sum(1 for f in done_flows if f.cls == "coflow")
        completed_bk = sum(1 for f in done_flows if f.cls == "bulk")
        
        # Milder completion rewards
        completed_reward = (completed_dl * 5.0) + (completed_cf * 2.0) + (completed_bk * 0.5)
        
        reward += completed_reward + utilization_reward'''

content = content.replace(old_reward, new_reward)

# 5. Fix _select_demands to avoid starvation
old_sort = '''        # Sort keys with priority: EDF for deadline flows, then coflow flows, then small flows, then by remaining bytes
        def sort_key(k):
            priority = 0

            # Highest priority for deadline flows with EDF
            if has_deadline.get(k, False):
                priority += 2000000000  # Base priority for deadline
                # EDF: earlier deadline = higher priority
                if k in earliest_deadline:
                    # Invert the deadline to make earlier deadlines have higher priority
                    # Add a large number to ensure it's positive
                    priority += (1000 - earliest_deadline[k]) * 100000  # EDF priority
                priority += deadline_count.get(k, 0) * 100000  # More deadline flows = higher priority
                priority += urgency_score.get(k, 0.0) * 1000  # Urgency factor  

            # Medium priority for coflow flows
            elif has_coflow.get(k, False):
                priority += 1000000000  # Base priority for coflow
                priority += coflow_count.get(k, 0) * 1000000  # More coflow flows = higher priority

            # High priority for small flows to prevent starvation
            if is_small_flow.get(k, False):
                priority += 500000000  # Additional priority for small flows    

            # Add remaining bytes as baseline
            priority += agg[k]  # Low priority for remaining bytes

            return priority'''

new_sort = '''        # Fairer sort key: uses smooth bonuses to prevent total starvation of bulk background traffic
        def sort_key(k):
            # Base priority: size of traffic demand (in roughly MB to keep it scaled nicely)
            score = agg[k] / 1e6
            
            # Soft priority multipliers, not massive absolute constants
            if has_deadline.get(k, False):
                score += 100.0  # Bumps it into the top list generally
                if k in earliest_deadline:
                    # Soft EDF boost
                    score += max(0, 100 - earliest_deadline[k] + self.t)
            elif has_coflow.get(k, False):
                score += 50.0  # Secondary importance
                
            if is_small_flow.get(k, False):
                score += 20.0
                
            return score'''
            
content = content.replace(old_sort, new_sort)

with open('env.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done refactoring env.py")