"""dead_node_elimination.py

Eliminate nodes unreachable from Input (dead nodes).
"""
from typing import Dict, List, Set
from automata.reachability_analysis import reachable_from


def eliminate_dead_nodes(adj: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Return a new adjacency dict with unreachable nodes removed."""
    reachable = reachable_from(adj, start="Input")
    new_adj = {k: [t for t in v if t in reachable] for k, v in adj.items() if k in reachable}
    return new_adj


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "Dead": ["X"], "ReLU": []}
    print("Before:", g)
    print("After:", eliminate_dead_nodes(g))
