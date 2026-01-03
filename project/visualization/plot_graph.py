"""plot_graph.py

Plot computational graphs (before/after) using networkx and matplotlib.
"""
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(adj: Dict[str, List[str]], path: str = "graph.png", title: str = "Graph") -> None:
    G = nx.DiGraph()
    for s, targets in adj.items():
        for t in targets:
            G.add_edge(s, t)
    
    plt.figure(figsize=(14, 6))
    
    # Custom layout: BFS layers from Input
    pos = {}
    try:
        # Distance from Input
        levels = nx.single_source_shortest_path_length(G, "Input")
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            level_groups.setdefault(level, []).append(node)
        
        # Assign coords: x = level, y = centered index in group
        for level, nodes in level_groups.items():
            count = len(nodes)
            for i, node in enumerate(nodes):
                # x grows right, y is centered
                pos[node] = (level * 2.0, (i - count / 2.0) * 1.5)
    except Exception:
        # Fallback if graph is disconnected or Input not found
        pos = nx.spring_layout(G, k=2.0)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=3000, alpha=0.9, edgecolors="black")
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowsize=25, edge_color="gray")
    # Draw labels with bounding box for readability
    labels = {n: n.replace("_", "\n") for n in G.nodes()} # split long names
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight="bold",
                           bbox=dict(facecolor="white", alpha=0.7, edgecolor='none', pad=1.0))
    
    plt.margins(0.1)
    plt.title(title, fontsize=16)
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    plot_graph(g, "graph_before.png", "Graph Before")
    print("Saved graph_before.png")