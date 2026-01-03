"""compare_graphs.py

Utility to create before/after graph images side-by-side (simple saving of two PNGs).
"""
from typing import Dict
from .plot_graph import plot_graph


def compare_graphs(adj_before: Dict[str, list], adj_after: Dict[str, list], prefix: str = "graph"):
    plot_graph(adj_before, f"{prefix}_before.png", "Graph Before")
    plot_graph(adj_after, f"{prefix}_after.png", "Graph After")
    print(f"Saved {prefix}_before.png and {prefix}_after.png")


if __name__ == "__main__":
    b = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    a = {"Input": ["FusedConvReLU"], "FusedConvReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    compare_graphs(b, a)