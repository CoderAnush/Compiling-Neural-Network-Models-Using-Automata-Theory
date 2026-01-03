"""graph_to_json.py

Helpers to export and import computational graph JSON files.
"""

from typing import Dict, List
import json


def save_graph(graph: Dict[str, List[str]], path: str) -> None:
    """Save adjacency list graph to JSON at `path`."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def load_graph(path: str) -> Dict[str, List[str]]:
    """Load adjacency list graph from JSON file at `path`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # quick smoke test
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    save_graph(g, "graph/demo_graph.json")
    print("Saved demo graph to graph/demo_graph.json")
