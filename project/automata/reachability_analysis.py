"""reachability_analysis.py

Utilities to perform reachability and dead-state detection on adjacency graphs (not just DFAs).
"""
from typing import Dict, List, Set


def reachable_from(adj: Dict[str, List[str]], start: str = "Input") -> Set[str]:
    """Return set of nodes reachable from `start` via DFS."""
    visited = set()
    stack = [start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        for t in adj.get(s, []):
            if t not in visited:
                stack.append(t)
    return visited


def dead_nodes(adj: Dict[str, List[str]], start: str = "Input") -> Set[str]:
    """Nodes not reachable from start are considered dead/unreachable."""
    r = reachable_from(adj, start)
    return set(adj.keys()) - r


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "Dead": ["X"], "MaxPool": []}
    print("Reachable:", reachable_from(g))
    print("Dead:", dead_nodes(g))
