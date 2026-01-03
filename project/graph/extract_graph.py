"""extract_graph.py

Utilities to extract a computational graph from a PyTorch model.

Approach:
- Use forward hooks to record the order in which modules are executed during a single forward pass
- Convert the execution order to a simple adjacency list where nodes are layer-type names
- Provide a helper to save the graph JSON

Functions:
- trace_execution_order(model, input_shape): returns ordered list of layer names recorded from a forward pass
- extract_adjacency_from_trace(trace): build adjacency list dict from ordered trace
- extract_graph(model, input_shape, names_map): end-to-end convenience function

Note: This module focuses on simple models (e.g., sequential CNNs used in this project). For complex dynamic control flow models more advanced tracing is required.
"""

from typing import List, Dict, Tuple, Optional
import torch
import json
import os
from collections import OrderedDict


def _short_name(module: torch.nn.Module) -> str:
    """Return a short readable name for a PyTorch module type.

    Examples: Conv2d -> Conv, ReLU -> ReLU, MaxPool2d -> MaxPool, Flatten -> Flatten, Linear -> Dense
    """
    cls = module.__class__.__name__
    mapping = {
        "Conv2d": "Conv",
        "ReLU": "ReLU",
        "MaxPool2d": "MaxPool",
        "Flatten": "Flatten",
        "Linear": "Dense",
        "LogSoftmax": "Softmax",
        "Softmax": "Softmax",
        "BatchNorm2d": "BatchNorm",
        "Dropout": "Dropout",
    }
    return mapping.get(cls, cls)


def trace_execution_order(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28)) -> List[str]:
    """Run a dummy input through `model` and record the sequence of module names executed.

    Args:
        model: PyTorch nn.Module to trace
        input_shape: shape for dummy input (batch, channels, H, W)

    Returns:
        ordered list of short layer names, including an initial 'Input' and a final 'Output' marker
    """
    hooks = []
    order = []

    def make_hook(name, module):
        def hook(module, input, output):
            # Combined name: ShortType_UniqueName (e.g., Conv_features_0)
            short = _short_name(module)
            # Sanitize dots to underscores for cleaner IR/filenames
            clean_uniq = name.replace(".", "_")
            order.append(f"{short}_{clean_uniq}")
        return hook

    # Register hooks on leaf modules (skip containers like Sequential)
    for name, module in model.named_modules():
        # avoid hooking the top-level model itself
        if module is model:
            continue
        # treat as leaf if it has no child modules
        if len(list(module.children())) == 0:
            try:
                hooks.append(module.register_forward_hook(make_hook(name, module)))
            except Exception:
                pass

    model.eval()
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        # mark start
        exec_order = ["Input"]
        _ = model(dummy)
        exec_order.extend(order)
        exec_order.append("Output")

    # remove hooks
    for h in hooks:
        h.remove()

    return exec_order


def extract_adjacency_from_trace(trace: List[str]) -> Dict[str, List[str]]:
    """Convert an ordered trace into an adjacency list (simple chain graph).

    Example:
        trace = ["Input", "Conv", "ReLU", "MaxPool", "Flatten", "Dense", "Softmax", "Output"]
        adjacency: {"Input": ["Conv"], "Conv": ["ReLU"], ...}
    """
    adj = OrderedDict()
    for i in range(len(trace) - 1):
        src = trace[i]
        dst = trace[i + 1]
        adj.setdefault(src, [])
        # avoid duplicates
        if dst not in adj[src]:
            adj[src].append(dst)
    # ensure last node exists in dict with empty list
    last = trace[-1]
    adj.setdefault(last, [])
    return adj


def extract_graph(model: torch.nn.Module, input_shape: Tuple[int, int, int, int] = (1, 1, 28, 28)) -> Dict[str, List[str]]:
    """End-to-end: trace model and return adjacency list.

    Args:
        model: torch.nn.Module
        input_shape: dummy input shape

    Returns:
        adjacency dict mapping node->list of next nodes
    """
    trace = trace_execution_order(model, input_shape)
    adj = extract_adjacency_from_trace(trace)
    return adj


def save_graph_json(graph: Dict[str, List[str]], path: str) -> None:
    """Save adjacency list as JSON file (pretty printed)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


if __name__ == "__main__":
    # quick demo when run directly
    from model.train_mnist import get_model

    print("Loading model (pretrained if available)...")
    m = get_model(pretrained=False)
    print("Tracing execution order...")
    trace = trace_execution_order(m)
    print("Trace:", trace)
    adj = extract_adjacency_from_trace(trace)
    print("Adjacency:")
    print(json.dumps(adj, indent=2))
    save_graph_json(adj, "graph/graph.json")
    print("Saved graph to graph/graph.json")
