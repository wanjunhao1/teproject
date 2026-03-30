# traffic.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

from real_data_loader import load_real_traffic


@dataclass
class Flow:
    flow_id: int
    cls: str  # "deadline" | "coflow" | "bulk"
    src: int
    dst: int
    size_bytes: float
    remaining_bytes: float
    arrival_t: int

    # deadline-only
    deadline_t: Optional[int] = None
    weight: float = 1.0

    # coflow-only
    coflow_id: Optional[int] = None


@dataclass
class Coflow:
    coflow_id: int
    flows: List[Flow] = field(default_factory=list)
    arrival_t: int = 0
    finished_t: Optional[int] = None

    def done(self) -> bool:
        return all(f.remaining_bytes <= 0 for f in self.flows)


class TrafficGenerator:
    """
    Replace with: trace replay / poisson arrivals / workload model.
    Produces new flows/coflows each time slot t.
    """

    def __init__(self, rng: np.random.Generator, num_nodes: int):
        self.rng = rng
        self.num_nodes = num_nodes
        self.next_flow_id = 0
        self.next_coflow_id = 0

    def sample_slot(self, t: int) -> Tuple[List[Flow], List[Coflow]]:
        new_flows: List[Flow] = []
        new_coflows: List[Coflow] = []

        # Increase traffic intensity by generating more flows
        # deadline flows
        for _ in range(3):  # Generate 3 deadline flows per slot on average
            if self.rng.random() < 0.6:
                src, dst = self._rand_pair()
                size = float(self.rng.uniform(10e6, 100e6))  # Larger flows
                ddl = t + int(self.rng.integers(2, 10))     # Tighter deadlines
                w = float(self.rng.uniform(0.5, 2.0))
                f = Flow(
                    flow_id=self._fid(),
                    cls="deadline",
                    src=src, dst=dst,
                    size_bytes=size,
                    remaining_bytes=size,
                    arrival_t=t,
                    deadline_t=ddl,
                    weight=w,
                )
                new_flows.append(f)

        # bulk flows
        for _ in range(4):  # Generate 4 bulk flows per slot on average
            if self.rng.random() < 0.5:
                src, dst = self._rand_pair()
                size = float(self.rng.uniform(50e6, 300e6))  # Larger flows
                f = Flow(
                    flow_id=self._fid(),
                    cls="bulk",
                    src=src, dst=dst,
                    size_bytes=size,
                    remaining_bytes=size,
                    arrival_t=t,
                )
                new_flows.append(f)

        # coflow: bundle of subflows
        for _ in range(2):  # Generate 2 coflows per slot on average
            if self.rng.random() < 0.4:
                cfid = self._cfid()
                n_sub = int(self.rng.integers(5, 15))  # More subflows per coflow
                flows = []
                for _ in range(n_sub):
                    src, dst = self._rand_pair()
                    size = float(self.rng.uniform(10e6, 120e6))  # Larger flows
                    flows.append(Flow(
                        flow_id=self._fid(),
                        cls="coflow",
                        src=src, dst=dst,
                        size_bytes=size,
                        remaining_bytes=size,
                        arrival_t=t,
                        coflow_id=cfid,
                    ))
                new_coflows.append(Coflow(coflow_id=cfid, flows=flows, arrival_t=t))

        return new_flows, new_coflows

    def _rand_pair(self) -> Tuple[int, int]:
        src = int(self.rng.integers(0, self.num_nodes))
        dst = int(self.rng.integers(0, self.num_nodes - 1))
        if dst >= src:
            dst += 1
        return src, dst

    def _fid(self) -> int:
        fid = self.next_flow_id
        self.next_flow_id += 1
        return fid

    def _cfid(self) -> int:
        cid = self.next_coflow_id
        self.next_coflow_id += 1
        return cid


class RealTrafficGenerator:
    """
    Load real traffic traces from datasets with dynamic burstiness support
    """

    def __init__(self, rng: np.random.Generator, num_nodes: int, trace_name: str = "facebook_hadoop", burstiness: float = 0.0):
        self.rng = rng
        self.num_nodes = num_nodes
        self.next_flow_id = 0
        self.next_coflow_id = 0
        self.trace_name = trace_name
        self.burstiness = burstiness  # Traffic burstiness factor (0-1)
        
        # Load real traffic data
        self.flow_data, self.coflow_data = load_real_traffic(trace_name, num_nodes)
        
        # Preprocess flows and coflows
        self.flows_by_time = {}
        self.coflows_by_time = {}
        
        for flow in self.flow_data:
            arrival_time = flow["arrival_time"]
            if arrival_time not in self.flows_by_time:
                self.flows_by_time[arrival_time] = []
            self.flows_by_time[arrival_time].append(flow)
        
        for coflow in self.coflow_data:
            arrival_time = coflow["arrival_time"]
            if arrival_time not in self.coflows_by_time:
                self.coflows_by_time[arrival_time] = []
            self.coflows_by_time[arrival_time].append(coflow)
        
        # For burstiness: track burst state
        self.in_burst = False
        self.burst_duration = 0
        self.burst_multiplier = 1.0

    def sample_slot(self, t: int) -> Tuple[List[Flow], List[Coflow]]:
        new_flows: List[Flow] = []
        new_coflows: List[Coflow] = []
        
        # Update burst state
        self._update_burst_state()
        
        # Get flows arriving at this time
        flow_map = {}  # Map original flow_id to Flow object
        if t in self.flows_by_time:
            for flow_data in self.flows_by_time[t]:
                # Apply burst multiplier to flow size
                size = flow_data["size"] * self.burst_multiplier
                flow = Flow(
                    flow_id=self._fid(),
                    cls=flow_data["cls"],
                    src=flow_data["src"],
                    dst=flow_data["dst"],
                    size_bytes=size,
                    remaining_bytes=size,
                    arrival_t=t,
                    deadline_t=flow_data.get("deadline"),
                    coflow_id=None  # Will be set later for coflow flows
                )
                new_flows.append(flow)
                flow_map[flow_data["flow_id"]] = flow
        
        # Get coflows arriving at this time
        if t in self.coflows_by_time:
            for coflow_data in self.coflows_by_time[t]:
                cfid = self._cfid()
                flows = []
                # Find corresponding flows
                for flow_id in coflow_data["flow_ids"]:
                    if flow_id in flow_map:
                        flow = flow_map[flow_id]
                        flow.coflow_id = cfid
                        flows.append(flow)
                # If no flows found, create new ones
                if not flows:
                    for flow_id in coflow_data["flow_ids"]:
                        # Find the flow in flow_data
                        for flow_data in self.flow_data:
                            if flow_data["flow_id"] == flow_id:
                                flow = Flow(
                                    flow_id=self._fid(),
                                    cls=flow_data["cls"],
                                    src=flow_data["src"],
                                    dst=flow_data["dst"],
                                    size_bytes=flow_data["size"],
                                    remaining_bytes=flow_data["size"],
                                    arrival_t=t,
                                    coflow_id=cfid
                                )
                                flows.append(flow)
                                new_flows.append(flow)
                                break
                if flows:
                    new_coflows.append(Coflow(coflow_id=cfid, flows=flows, arrival_t=t))
        
        return new_flows, new_coflows

    def _update_burst_state(self):
        """Update traffic burst state"""
        if self.burstiness <= 0:
            return
        
        # If in burst, decrement duration
        if self.in_burst:
            self.burst_duration -= 1
            if self.burst_duration <= 0:
                self.in_burst = False
                self.burst_multiplier = 1.0
        else:
            # Randomly start a burst
            if self.rng.random() < self.burstiness * 0.1:  # 10% chance per slot when burstiness=1
                self.in_burst = True
                self.burst_duration = self.rng.integers(5, 20)  # Burst lasts 5-20 slots
                self.burst_multiplier = self.rng.uniform(1.5, 3.0)  # 1.5x to 3x traffic

    def _fid(self) -> int:
        fid = self.next_flow_id
        self.next_flow_id += 1
        return fid

    def _cfid(self) -> int:
        cid = self.next_coflow_id
        self.next_coflow_id += 1
        return cid
