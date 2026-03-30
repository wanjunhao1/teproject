# common.py
from dataclasses import dataclass
from typing import List

@dataclass
class Path:
    links: List[int]
    prop_delay: float = 0.0  # optional (sum of link delays)
