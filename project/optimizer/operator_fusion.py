"""operator_fusion.py

Detect patterns (Conv -> ReLU) and fuse them into a single FusedConvReLU node.
This module performs simple pattern matching on linear chain graphs.
"""
from typing import Dict, List, Tuple


def fuse_conv_relu(adj: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Fuse Conv->ReLU patterns in `adj`.

    Returns (new_adj, fused_pairs)
    """
    fused = []
    new_adj = {k: list(v) for k, v in adj.items()}  # shallow copy

    nodes = list(adj.keys())
    i = 0
    while i < len(nodes) - 1:
        a = nodes[i]
        b = nodes[i + 1]
        if a.startswith("Conv") and b.startswith("ReLU"):
            # perform fusion: replace Conv and ReLU with FusedConvReLU in graph
            # e.g. Conv_features_0 + ReLU_features_1 -> FusedConvReLU_features_0
            suffix = a.split("_", 1)[1] if "_" in a else ""
            fused_name = f"FusedConvReLU_{suffix}" if suffix else "FusedConvReLU"

            # find predecessors of 'Conv' and successors of 'ReLU'
            preds = [p for p, t in adj.items() if a in t]
            succs = adj.get(b, [])
            # remove a and b
            for p in preds:
                new_adj[p] = [fused_name if x == a else x for x in new_adj.get(p, [])]
            new_adj[fused_name] = list(succs)
            if a in new_adj:
                del new_adj[a]
            if b in new_adj:
                del new_adj[b]
            fused.append(f"{a}+{b}")
            # update nodes list to include fused_name in place of a,b
            nodes[i] = fused_name
            del nodes[i + 1]
            # do not advance i to catch consecutive fusions
        else:
            i += 1
    return new_adj, fused


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": []}
    new_g, fused = fuse_conv_relu(g)
    print("Fused:", fused)
    print("New graph:", new_g)