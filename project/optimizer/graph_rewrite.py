"""graph_rewrite.py

Provide utilities to rewrite graphs using simple pattern rules.

Example rewrite rule: Replace Flatten->Dense with ReshapeDense (demonstrative)
"""
from typing import Dict, List, Tuple


def rewrite_flatten_dense(adj: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[str]]:
    """Rewrite Flatten->Dense -> ReshapeDense"""
    new_adj = {k: list(v) for k, v in adj.items()}
    rewritten = []
    nodes = list(adj.keys())
    for i in range(len(nodes) - 1):
        a = nodes[i]
        b = nodes[i + 1]
        if a.startswith("Flatten") and b.startswith("Dense"):
            suffix = b.split("_", 1)[1] if "_" in b else ""
            new_name = f"ReshapeDense_{suffix}" if suffix else "ReshapeDense"
            preds = [p for p, t in adj.items() if a in t]
            succs = adj.get(b, [])
            for p in preds:
                new_adj[p] = [new_name if x == a else x for x in new_adj.get(p, [])]
            new_adj[new_name] = list(succs)
            if a in new_adj:
                del new_adj[a]
            if b in new_adj:
                del new_adj[b]
            rewritten.append(f"{a}+{b}->{new_name}")
    return new_adj, rewritten


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["Flatten"], "Flatten": ["Dense"], "Dense": []}
    new_g, rew = rewrite_flatten_dense(g)
    print("Rewritten:", rew)
    print("New graph:", new_g)