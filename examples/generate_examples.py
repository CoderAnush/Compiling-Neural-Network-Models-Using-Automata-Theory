"""Generate example visualizations for the README and examples folder.
This script will create `examples/images/graph_before.png` and `examples/images/graph_after.png`.
"""
import sys
import os
# Ensure the 'project' folder is importable when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "project"))
from visualization.plot_graph import plot_graph
from optimizer.optimize_all import optimize_graph


def main():
    adj = {
        "Input": ["Conv_features_0"],
        "Conv_features_0": ["ReLU_features_1"],
        "ReLU_features_1": ["MaxPool_features_2"],
        "MaxPool_features_2": ["Conv_features_3"],
        "Conv_features_3": ["ReLU_features_4"],
        "ReLU_features_4": ["MaxPool_features_5"],
        "MaxPool_features_5": ["Flatten_features_6"],
        "Flatten_features_6": ["Dense_classifier_0"],
        "Dense_classifier_0": ["ReLU_classifier_1"],
        "ReLU_classifier_1": ["Dense_classifier_2"],
        "Dense_classifier_2": ["Softmax_classifier_3"],
        "Softmax_classifier_3": ["Output"],
        "Output": [],
    }

    plot_graph(adj, "examples/images/graph_before.png", "Graph Before (example)")
    opt = optimize_graph(adj)
    plot_graph(opt, "examples/images/graph_after.png", "Graph After (example)")
    print("Examples generated: examples/images/graph_before.png, examples/images/graph_after.png")


if __name__ == "__main__":
    main()
