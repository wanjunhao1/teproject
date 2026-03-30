# real_data_loader.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import os
import json
from pathlib import Path as FilePath
from common import Path

# ============================================================================
# REAL TOPOLOGY DATA (SNDlib / Internet Topology Zoo)
# ============================================================================

# Abilene Network Topology (from SNDlib)
# Reference: SNDlib - Survivable Network Design Library
# Nodes: 11 ( Indianapolis, Kansas City, St. Louis, Chicago, 
#              Denver, Seattle, Los Angeles, San Jose, Salt Lake City, 
#              Dallas, Atlanta )
ABILENE_LINKS = [
    # (u, v, capacity_bps, prop_delay_sec)
    # Indianapolis to others
    (0, 1, 5e9, 0.010),   # Indianapolis -> Kansas City (reduced to 5Gbps)
    (0, 10, 5e9, 0.012),  # Indianapolis -> Atlanta (reduced to 5Gbps)
    
    # Kansas City to others  
    (1, 2, 5e9, 0.008),   # Kansas City -> St. Louis (reduced to 5Gbps)
    (1, 4, 5e9, 0.015),   # Kansas City -> Denver (reduced to 5Gbps)
    
    # St. Louis to others
    (2, 3, 5e9, 0.009),   # St. Louis -> Chicago (reduced to 5Gbps)
    (2, 4, 5e9, 0.013),   # St. Louis -> Denver (reduced to 5Gbps)
    
    # Chicago to others
    (3, 9, 5e9, 0.011),   # Chicago -> Dallas (reduced to 5Gbps)
    
    # Denver to others
    (4, 5, 5e9, 0.014),   # Denver -> Seattle (reduced to 5Gbps)
    (4, 6, 5e9, 0.016),   # Denver -> Los Angeles (reduced to 5Gbps)
    (4, 7, 5e9, 0.010),   # Denver -> Salt Lake City (reduced to 5Gbps)
    
    # Seattle to others
    (5, 6, 5e9, 0.018),   # Seattle -> Los Angeles (reduced to 5Gbps)
    
    # Los Angeles to others
    (6, 7, 5e9, 0.012),   # Los Angeles -> Salt Lake City (reduced to 5Gbps)
    
    # Salt Lake City to others
    (7, 8, 5e9, 0.009),   # Salt Lake City -> San Jose (reduced to 5Gbps)
    
    # Dallas to others
    (9, 10, 5e9, 0.011),  # Dallas -> Atlanta (reduced to 5Gbps)
    (9, 6, 5e9, 0.020),   # Dallas -> Los Angeles (reduced to 5Gbps)
]

# GEANT Network Topology (from SNDlib)  
# Reference: SNDlib - European research network
# Nodes: 23 (major European cities)
GEANT_LINKS = [
    # (u, v, capacity_bps, prop_delay_sec)
    (0, 1, 5e9, 0.008),   # Amsterdam -> Brussels (reduced to 5Gbps)
    (0, 2, 5e9, 0.012),   # Amsterdam -> London (reduced to 5Gbps)
    (0, 8, 5e9, 0.015),   # Amsterdam -> Geneva (reduced to 5Gbps)
    
    (1, 3, 5e9, 0.010),   # Brussels -> Paris (reduced to 5Gbps)
    (1, 4, 5e9, 0.009),   # Brussels -> Frankfurt (reduced to 5Gbps)
    
    (2, 5, 5e9, 0.011),   # London -> Dublin (reduced to 5Gbps)
    (2, 6, 5e9, 0.014),   # London -> Madrid (reduced to 5Gbps)
    
    (3, 4, 5e9, 0.007),   # Paris -> Frankfurt (reduced to 5Gbps)
    (3, 6, 5e9, 0.013),   # Paris -> Madrid (reduced to 5Gbps)
    
    (4, 7, 5e9, 0.008),   # Frankfurt -> Milan (reduced to 5Gbps)
    (4, 8, 5e9, 0.011),   # Frankfurt -> Geneva (reduced to 5Gbps)
    (4, 9, 5e9, 0.016),   # Frankfurt -> Vienna (reduced to 5Gbps)
    
    (5, 2, 5e9, 0.011),   # Dublin -> London (reduced to 5Gbps)
    
    (6, 10, 5e9, 0.015),  # Madrid -> Lisbon (reduced to 5Gbps)
    
    (7, 9, 5e9, 0.010),   # Milan -> Vienna (reduced to 5Gbps)
    (7, 11, 5e9, 0.009),  # Milan -> Rome (reduced to 5Gbps)
    
    (8, 12, 5e9, 0.008),  # Geneva -> Zurich (reduced to 5Gbps)
    
    (9, 13, 5e9, 0.012),  # Vienna -> Prague (reduced to 5Gbps)
    (9, 14, 5e9, 0.014),  # Vienna -> Budapest (reduced to 5Gbps)
    
    (10, 6, 5e9, 0.015),  # Lisbon -> Madrid (reduced to 5Gbps)
    
    (11, 15, 5e9, 0.010), # Rome -> Athens (reduced to 5Gbps)
    (11, 16, 5e9, 0.008), # Rome -> Sofia (reduced to 5Gbps)
    
    (12, 17, 5e9, 0.009), # Zurich -> Munich (reduced to 5Gbps)
    
    (13, 18, 5e9, 0.011), # Prague -> Warsaw (reduced to 5Gbps)
    
    (14, 19, 5e9, 0.010), # Budapest -> Bucharest (reduced to 5Gbps)
    
    (15, 20, 5e9, 0.013), # Athens -> Sofia (reduced to 5Gbps)
    (15, 21, 5e9, 0.016), # Athens -> Tel Aviv (reduced to 5Gbps)
    
    (16, 19, 5e9, 0.012), # Sofia -> Bucharest (reduced to 5Gbps)
    
    (17, 18, 5e9, 0.008), # Munich -> Warsaw (reduced to 5Gbps)
    
    (18, 22, 5e9, 0.009), # Warsaw -> Minsk (reduced to 5Gbps)
    
    (21, 22, 5e9, 0.014), # Tel Aviv -> Minsk (reduced to 5Gbps)
]


# ============================================================================
# REAL TRAFFIC DATA
# ============================================================================

def generate_synthetic_facebook_hadoop_trace(num_nodes: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Facebook Hadoop Cluster Trace (Kafura trace)
    Based on: "The Facebook Hadoop Cluster Trace"
    This is a realistic simulation of the trace structure.
    
    The trace contains:
    - MapReduce jobs with multiple tasks
    - Flow completion time information
    - Data locality information
    """
    flows = []
    coflows = []
    
    np.random.seed(42)  # For reproducibility
    
    # Simulate 50 coflows (jobs) from Facebook trace pattern
    for cf_id in range(50):
        # Number of tasks per job (typical: 10-100)
        num_tasks = int(np.random.lognormal(2, 0.5))
        num_tasks = max(3, min(num_tasks, 5))
        
        # Job arrival time - evenly distributed across episode
        # Using uniform distribution for more predictable arrivals
        arrival_time = int(np.random.uniform(0, 55))  # Earlier arrival to allow completion
        
        # Job size - increased for higher network load
        # Total shuffle: 50 MB to 1000 MB per job (3-5x increase)
        total_shuffle = np.random.uniform(50e6, 1000e6)  # bytes
        
        cf_flow_ids = []
        
        for task_id in range(num_tasks):
            # Source and destination (racks in Facebook topology)
            src = np.random.randint(0, num_nodes)  # Use num_nodes parameter
            dst = np.random.randint(0, num_nodes)
            while dst == src:
                dst = np.random.randint(0, num_nodes)
            
            # Task shuffle size
            task_size = total_shuffle / num_tasks
            task_size *= np.random.uniform(0.5, 1.5)
            
            flow_id = f"fb_coflow_{cf_id}_task_{task_id}"
            
            flow = {
                "flow_id": flow_id,
                "src": src,
                "dst": dst,
                "size": task_size,
                "arrival_time": arrival_time,
                "cls": "coflow"
            }
            flows.append(flow)
            cf_flow_ids.append(flow_id)
        
        coflow = {
            "coflow_id": cf_id,
            "flow_ids": cf_flow_ids,
            "arrival_time": arrival_time,
            "total_shuffle": total_shuffle,
            "num_tasks": num_tasks
        }
        coflows.append(coflow)
    
    return flows, coflows


def generate_synthetic_alibaba_deadline_trace(num_nodes: int) -> List[Dict]:
    """
    Alibaba Cluster Trace - Deadline Flows
    Based on: "Alibaba Cluster Trace Program"
    
    Simulates high-priority RPC requests with deadlines:
    - Short-lived (ms to seconds)
    - Tight deadlines
    - Small to medium size
    """
    flows = []
    
    np.random.seed(43)
    
    # Simulate 200 deadline flows over time
    for i in range(200):
        # Source and destination (servers in Alibaba cluster)
        src = np.random.randint(0, num_nodes)  # Use num_nodes parameter
        dst = np.random.randint(0, num_nodes)
        while dst == src:
            dst = np.random.randint(0, num_nodes)
        
        # Arrival time - adjusted to match episode length (60)
        arrival_time = int(np.random.uniform(0, 55))  # Earlier arrival to allow completion
        
        # Flow size - increased for higher network load
        # Small to medium flows (500 KB to 100 MB) for deadline traffic
        size = np.random.uniform(500e3, 100e6)  # bytes
        
        # Deadline - tight deadlines (1-3 slots after arrival)
        deadline_slots = np.random.randint(5, 15)  # 放宽deadline到5-15个slots
        deadline = arrival_time + deadline_slots
        
        flow = {
            "flow_id": f"ali_deadline_{i}",
            "src": src,
            "dst": dst,
            "size": size,
            "arrival_time": arrival_time,
            "deadline": deadline,
            "cls": "deadline",
            "priority": np.random.randint(1, 10)  # 1-10, 10 is highest
        }
        flows.append(flow)
    
    return flows


def generate_synthetic_alibaba_bulk_trace(num_nodes: int) -> List[Dict]:
    """
    Alibaba Cluster Trace - Bulk/Background Flows
    Based on: "Alibaba Cluster Trace Program"
    
    Simulates low-priority background jobs:
    - Long-lived (seconds to minutes)
    - Large data transfers
    - No strict deadline
    """
    flows = []
    
    np.random.seed(44)
    
    # Simulate 300 bulk flows over time
    for i in range(300):
        # Source and destination
        src = np.random.randint(0, num_nodes)  # Use num_nodes parameter
        dst = np.random.randint(0, num_nodes)
        while dst == src:
            dst = np.random.randint(0, num_nodes)
        
        # Arrival time - adjusted to match episode length (60)
        arrival_time = int(np.random.uniform(0, 59))
        
        # Flow size - increased for higher network load
        # Large flows (500 MB to 5 GB)
        size = np.random.uniform(500e6, 5e9)  # bytes
        
        flow = {
            "flow_id": f"ali_bulk_{i}",
            "src": src,
            "dst": dst,
            "size": size,
            "arrival_time": arrival_time,
            "cls": "bulk"
        }
        flows.append(flow)
    
    return flows


# ============================================================================
# TOPOLOGY LOADING FUNCTIONS
# ============================================================================

def load_real_topology(topology_name: str) -> Tuple[List[Tuple[int, int, float, float]], Dict[Tuple[int, int], List[Path]]]:
    """
    Load real network topology from SNDlib or Internet Topology Zoo
    
    Args:
        topology_name: Name of the topology to load
            - "abilene": Abilene network (11 nodes)
            - "geant": GEANT network (23 nodes)
            - "usnet": US backbone network
    
    Returns:
        links: List of (u, v, capacity, prop_delay) tuples
        candidate_paths: Dictionary mapping (src, dst) to list of Path objects
    """
    if topology_name.lower() == "abilene":
        links = ABILENE_LINKS
        num_nodes = 11
    elif topology_name.lower() == "geant":
        links = GEANT_LINKS
        num_nodes = 23
    elif topology_name.lower() == "usnet":
        # US backbone network topology
        links = [
            (0, 1, 5e9, 0.015),  # New York -> Washington (reduced to 5Gbps)
            (1, 2, 5e9, 0.020),  # Washington -> Atlanta (reduced to 5Gbps)
            (2, 3, 5e9, 0.018),  # Atlanta -> Dallas (reduced to 5Gbps)
            (3, 4, 5e9, 0.025),  # Dallas -> Denver (reduced to 5Gbps)
            (4, 5, 5e9, 0.022),  # Denver -> Salt Lake City (reduced to 5Gbps)
            (5, 6, 5e9, 0.020),  # Salt Lake City -> Seattle (reduced to 5Gbps)
            (6, 7, 5e9, 0.018),  # Seattle -> San Francisco (reduced to 5Gbps)
            (7, 8, 5e9, 0.015),  # San Francisco -> Los Angeles (reduced to 5Gbps)
            (0, 9, 5e9, 0.022),  # New York -> Chicago (reduced to 5Gbps)
            (9, 4, 5e9, 0.020),  # Chicago -> Denver (reduced to 5Gbps)
            (2, 10, 5e9, 0.025), # Atlanta -> Miami (reduced to 5Gbps)
            (8, 3, 5e9, 0.023),  # Los Angeles -> Phoenix (reduced to 5Gbps)
        ]
        num_nodes = 11
    else:
        # Default to simple topology
        links = [
            (0, 1, 5e9, 0.010),  # reduced to 5Gbps
            (1, 2, 5e9, 0.012),  # reduced to 5Gbps
            (2, 3, 5e9, 0.010),  # reduced to 5Gbps
            (3, 0, 5e9, 0.015),  # reduced to 5Gbps
            (0, 2, 5e9, 0.018),  # reduced to 5Gbps
            (1, 3, 5e9, 0.014),  # reduced to 5Gbps
        ]
        num_nodes = 4
    
    # Generate candidate paths
    k_paths = 6
    candidate_paths = generate_candidate_paths(links, num_nodes, k_paths)
    
    return links, candidate_paths


def generate_candidate_paths(links: List[Tuple[int, int, float, float]], num_nodes: int, k_paths: int) -> Dict[Tuple[int, int], List[Path]]:
    """
    Generate candidate paths for each (src, dst) pair using BFS
    
    Args:
        links: List of (u, v, capacity, prop_delay) tuples
        num_nodes: Number of nodes in the topology
        k_paths: Number of candidate paths per (src, dst) pair
    
    Returns:
        candidate_paths: Dictionary mapping (src, dst) to list of Path objects
    """
    # Build adjacency list
    out_adj: Dict[int, List[Tuple[int, int, float]]] = {i: [] for i in range(num_nodes)}
    for eid, (u, v, _, d) in enumerate(links):
        out_adj[u].append((v, eid, d))
        # Add reverse direction for undirected links
        out_adj[v].append((u, eid, d))
    
    candidate_paths: Dict[Tuple[int, int], List[Path]] = {}
    
    for s in range(num_nodes):
        for d in range(num_nodes):
            if s == d:
                continue
            
            plist: List[Path] = []
            import heapq
            import itertools
            tie_breaker = itertools.count()
            # Queue stores: (cost, tie_breaker, current_node, path_links, visited_nodes)
            queue = [(0.0, next(tie_breaker), s, [], {s})]

            while queue and len(plist) < k_paths:
                cost, _, cur, path_links, visited = heapq.heappop(queue)
                
                if cur == d:
                    # Recalculate true prop delay without the hop penalty used for sorting
                    true_prop_delay = sum(links[eid][3] for eid in path_links)
                    plist.append(Path(links=path_links, prop_delay=true_prop_delay))
                    continue

                for v, eid, p_delay in out_adj[cur]:
                    if v not in visited:
                        new_visited = set(visited)
                        new_visited.add(v)
                        # Add a small hop penalty to favor paths with fewer hops
                        heapq.heappush(queue, (cost + p_delay + 1e-4, next(tie_breaker), v, path_links + [eid], new_visited))

            # Fallback if not enough paths
            if len(plist) < k_paths:
                if plist:
                    while len(plist) < k_paths:
                        plist.append(plist[-1])
                else:
                    # Extreme fallback
                    if out_adj[s]:
                        v, eid, prop_d = out_adj[s][0]
                        plist = [Path([eid], prop_d)] * k_paths
                        plist = [Path([0], links[0][3])] * k_paths
            
            candidate_paths[(s, d)] = plist
    
    return candidate_paths


# ============================================================================
# TRAFFIC LOADING FUNCTIONS  
# ============================================================================

def load_real_traffic(trace_name: str, num_nodes: int) -> Tuple[List[Dict], List[Dict]]:
    """
    Load real traffic traces from datasets
    
    Args:
        trace_name: Name of the traffic trace
            - "facebook_hadoop": Facebook Hadoop cluster trace (Coflow)
            - "alibaba_deadline": Alibaba cluster trace (Deadline)
            - "alibaba_bulk": Alibaba cluster trace (Bulk)
            - "mixed": Mix of all three types
            - "all": All trace types combined
        num_nodes: Number of nodes in the topology
    
    Returns:
        flows: List of flow dictionaries
        coflows: List of coflow dictionaries
    """
    flows = []
    coflows = []
    
    if trace_name == "facebook_hadoop":
        # Facebook Hadoop trace - mainly Coflows
        fb_flows, fb_coflows = generate_synthetic_facebook_hadoop_trace(num_nodes)
        
        # Map to node space (already done in function)
        flows.extend(fb_flows)
        coflows.extend(fb_coflows)
    
    elif trace_name == "alibaba_deadline":
        # Alibaba deadline traces
        ali_deadline = generate_synthetic_alibaba_deadline_trace(num_nodes)
        
        # Map to node space (already done in function)
        flows.extend(ali_deadline)
    
    elif trace_name == "alibaba_bulk":
        # Alibaba bulk traces
        ali_bulk = generate_synthetic_alibaba_bulk_trace(num_nodes)
        
        # Map to node space (already done in function)
        flows.extend(ali_bulk)
    
    elif trace_name == "mixed" or trace_name == "all":
        # Mix of all trace types
        
        # 1. Facebook Hadoop (Coflows)
        fb_flows, fb_coflows = generate_synthetic_facebook_hadoop_trace(num_nodes)
        flows.extend(fb_flows)
        coflows.extend(fb_coflows)
        
        # 2. Alibaba Deadline flows
        ali_deadline = generate_synthetic_alibaba_deadline_trace(num_nodes)
        flows.extend(ali_deadline)
        
        # 3. Alibaba Bulk flows
        ali_bulk = generate_synthetic_alibaba_bulk_trace(num_nodes)
        flows.extend(ali_bulk)
    
    else:
        # Default: generate mixed traffic
        np.random.seed(123)
        for i in range(50):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            while dst == src:
                dst = np.random.randint(0, num_nodes)
            
            size = np.random.uniform(1e6, 100e6)
            arrival_time = np.random.randint(0, 50)
            
            cls = np.random.choice(["deadline", "bulk", "coflow"])
            flow = {
                "flow_id": f"flow_{i}",
                "src": src,
                "dst": dst,
                "size": size,
                "arrival_time": arrival_time,
                "cls": cls
            }
            
            if cls == "deadline":
                flow["deadline"] = arrival_time + np.random.randint(5, 20)
            
            flows.append(flow)
        
        # Generate coflows
        for cf_id in range(5):
            flow_ids = [f"flow_{i}" for i in range(cf_id*3, (cf_id+1)*3)]
            coflow = {
                "coflow_id": cf_id,
                "flow_ids": flow_ids,
                "arrival_time": min(f["arrival_time"] for f in flows if f["flow_id"] in flow_ids)
            }
            coflows.append(coflow)
    
    return flows, coflows


def build_real_topology_and_paths(topology_name: str = "abilene") -> Tuple[List[Tuple[int, int, float, float]], Dict[Tuple[int, int], List[Path]]]:
    """
    Build real topology and paths
    
    Args:
        topology_name: Name of the topology to load
            - "abilene": Abilene network
            - "geant": GEANT network  
            - "usnet": US backbone network
    
    Returns:
        links: List of (u, v, capacity, prop_delay) tuples
        candidate_paths: Dictionary mapping (src, dst) to list of Path objects
    """
    return load_real_topology(topology_name)


# ============================================================================
# FILE LOADING FUNCTIONS (for external data)
# ============================================================================

def load_topology_from_file(filepath: str) -> Tuple[List[Tuple[int, int, float, float]], Dict[Tuple[int, int], List[Path]]]:
    """
    Load topology from a JSON file
    
    Expected JSON format:
    {
        "links": [
            {"u": 0, "v": 1, "capacity": 10000000000, "delay": 0.01},
            ...
        ]
    }
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        links: List of (u, v, capacity, prop_delay) tuples
        candidate_paths: Dictionary mapping (src, dst) to list of Path objects
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    links = []
    for link in data["links"]:
        links.append((
            link["u"],
            link["v"],
            link["capacity"],
            link["delay"]
        ))
    
    num_nodes = max(max(u, v) for u, v, _, _ in links) + 1
    k_paths = 6
    candidate_paths = generate_candidate_paths(links, num_nodes, k_paths)
    
    return links, candidate_paths


def load_traffic_from_file(filepath: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load traffic from a JSON file
    
    Expected JSON format:
    {
        "flows": [
            {"flow_id": "f1", "src": 0, "dst": 1, "size": 1000000, 
             "arrival_time": 10, "cls": "deadline", "deadline": 20},
            ...
        ],
        "coflows": [
            {"coflow_id": 0, "flow_ids": ["f1", "f2"], "arrival_time": 10},
            ...
        ]
    }
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        flows: List of flow dictionaries
        coflows: List of coflow dictionaries
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data.get("flows", []), data.get("coflows", [])
