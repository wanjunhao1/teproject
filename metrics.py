# metrics.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
from traffic import Flow, Coflow


def compute_mlu(link_load: np.ndarray, link_capacity: np.ndarray, eps: float = 1e-9) -> float:
    util = link_load / (link_capacity + eps)
    return float(np.max(util)) if util.size else 0.0


def compute_deadline_penalty(done_flows: List[Flow], active_flows: List[Flow], finished_t: int) -> Dict[str, float]:
    """
    Lateness-based penalty (weighted).
    Computes both the final lateness penalty for newly finished flows,
    and a real-time lateness penalty for flows that are still active but have missed their deadline.
    """
    pen = 0.0
    miss = 0
    total = 0
    
    # 1. Final penalty for done flows
    for f in done_flows:
        if f.cls != "deadline":
            continue
        total += 1
        if f.deadline_t is None:
            continue
        lateness = max(0, finished_t - f.deadline_t)
        if lateness > 0:
            miss += 1
        pen += 0.1 * float(f.weight) * float(lateness)
        
    # 2. Real-time active penalty for flows that are suffering
    for f in active_flows:
        if f.cls == "deadline" and f.deadline_t is not None:
            if finished_t > f.deadline_t:
                lateness = finished_t - f.deadline_t
                # 放大单步延迟的惩罚，采用平方级别或较大倍数：严重延时时遭受毁灭性扣分
                pen += 0.1 * float(f.weight) * float(lateness)

    return {"deadline_penalty": float(pen), "deadline_miss": float(miss), "deadline_total": float(total)}


def compute_coflow_cct(done_coflows: List[Coflow]) -> Dict[str, float]:
    """
    Average coflow completion time over completed coflows in this slot.
    CCT for a coflow = finished_t - arrival_t
    """
    if not done_coflows:
        return {"avg_coflow_cct": 0.0, "num_coflows_done": 0.0}

    ccts = []
    for cf in done_coflows:
        if cf.finished_t is None:
            continue
        # Ensure at least 1 time slot
        cct = max(1.0, float(cf.finished_t - cf.arrival_t))
        ccts.append(cct)

    if not ccts:
        return {"avg_coflow_cct": 0.0, "num_coflows_done": 0.0}

    avg_cct = float(np.mean(ccts))
    return {"avg_coflow_cct": avg_cct, "num_coflows_done": float(len(ccts))}
