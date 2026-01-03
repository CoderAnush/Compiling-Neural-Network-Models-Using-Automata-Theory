"""demo_output.py

One-click demo that runs the full pipeline:
- load or train model
- extract computational graph
- build DFA & reachability
- generate IR
- optimize graph
- run original and optimized models
- benchmark
- generate visualizations
- print outputs and save report
"""
from model.train_mnist import get_model
from graph.extract_graph import extract_graph, save_graph_json
from automata.dfa_builder import build_dfa_from_adjacency, reachability
from ir.generate_ir import graph_to_ir, ir_as_list
from ir.ir_printer import pretty_print_ir, pretty_print_structured
from optimizer.optimize_all import optimize_graph
from executor.run_original import run_original
from executor.run_optimized import run_optimized
from benchmark.benchmark_compare import benchmark
from visualization.plot_graph import plot_graph

try:
    from visualization.draw_dfa import draw_dfa
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    print("Warning: graphviz not available, DFA diagrams will be skipped")

from visualization.compare_graphs import compare_graphs
import json
import os


REPORT_PATH = "report.md"


def save_report(content: str, path: str = REPORT_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def run_demo():
    total_steps = 9
    
    # 1. load model
    print(f"[1/{total_steps}] (11%) Loading model...")
    model = get_model(pretrained=False)
    print("✓ Model loaded")

    # 2. extract graph
    print(f"\n[2/{total_steps}] (22%) Extracting computational graph...")
    adj = extract_graph(model)
    save_graph_json(adj, "graph/graph_original.json")
    print("✓ Graph extracted and saved")

    # 3. DFA
    print(f"\n[3/{total_steps}] (33%) Building DFA and analyzing reachability...")
    dfa = build_dfa_from_adjacency(adj)
    reachable, unreachable = reachability(dfa)
    print(f"✓ DFA built. Reachable: {len(reachable)}, Unreachable: {len(unreachable)}")

    # 4. IR
    print(f"\n[4/{total_steps}] (44%) Generating IR (Intermediate Representation)...")
    ir_lines = graph_to_ir(adj)
    ir_struct = ir_as_list(adj)
    print(f"✓ IR generated ({len(ir_lines)} instructions)")

    # 5. Optimize
    print(f"\n[5/{total_steps}] (55%) Optimizing graph (fusion + rewrites)...")
    opt = optimize_graph(adj)
    save_graph_json(opt, "graph/graph_optimized.json")
    print(f"✓ Optimization complete. Nodes: {len(adj)} → {len(opt)}")

    # 6. Run original & optimized using real MNIST test data
    print(f"\n[6/{total_steps}] (66%) Running inference (original & optimized) on real MNIST test data...")
    print("Downloading/loading MNIST test set (if not cached)...")
    try:
        out_orig, t_orig = run_original(use_real_data=True)
        out_opt, t_opt = run_optimized(opt, use_real_data=True)
        print(f"✓ Inference complete. Original: {t_orig:.2f}ms, Optimized: {t_opt:.2f}ms")
    except Exception as e:
        print(f"Error during MNIST download or inference: {e}")
        print("Falling back to random input to finish demo; re-run with a stable network to use real data.")
        out_orig, t_orig = run_original(use_real_data=False)
        out_opt, t_opt = run_optimized(opt, use_real_data=False)
        print(f"✓ Inference (random fallback) complete. Original: {t_orig:.2f}ms, Optimized: {t_opt:.2f}ms")

    # 7. Benchmark (we'll run the benchmark which currently runs on random batches by default)
    print(f"\n[7/{total_steps}] (77%) Running benchmark (1 iteration)...")
    bench = benchmark(adj, opt, runs=1)
    print("✓ Benchmark complete")

    # 8. Visualize
    print(f"\n[8/{total_steps}] (88%) Generating visualizations...")
    plot_graph(adj, "visualizations/graph_before.png", "Graph Before")
    plot_graph(opt, "visualizations/graph_after.png", "Graph After")
    if HAS_GRAPHVIZ:
        draw_dfa(dfa.states, dfa.transitions, dfa.start, dfa.accept, "visualizations/dfa1", "DFA Before")
        print("✓ Visualizations saved (graphs + DFA)")
    else:
        print("✓ Graph visualizations saved (DFA skipped - graphviz not installed)")

    # 9. Report
    print(f"\n[9/{total_steps}] (100%) Generating final report...")
    report = [
        "# Neural Network Compiler Demo Report\n",
        "## Graph (original)\n",
        json.dumps(adj, indent=2),
        "\n\n## IR (original)\n",
        pretty_print_ir(ir_lines),
        "\n\n## Graph (optimized)\n",
        json.dumps(opt, indent=2),
        "\n\n## Benchmark\n",
        json.dumps(bench, indent=2),
    ]
    save_report("\n".join(report))
    print("✓ Report saved\n")
    print("="*50)
    print("Demo finished. Report saved to report.md and visualizations saved to visualizations/ folder.")
    print("="*50)


if __name__ == "__main__":
    run_demo()
