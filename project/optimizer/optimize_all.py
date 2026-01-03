"""optimize_all.py

Run the full optimization pipeline on an adjacency graph.
"""
from typing import Dict, List, Tuple
from optimizer.dead_node_elimination import eliminate_dead_nodes
from optimizer.operator_fusion import fuse_conv_relu
from optimizer.graph_rewrite import rewrite_flatten_dense


def optimize_graph(adj: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Apply sequential optimization passes and return optimized graph."""
    g1 = eliminate_dead_nodes(adj)
    g2, fused = fuse_conv_relu(g1)
    g3, rewritten = rewrite_flatten_dense(g2)
    # could add constant folding etc
    return g3


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    print("Original:", g)
    print("Optimized:", optimize_graph(g))