"""benchmark_compare.py

Benchmark original vs optimized model in terms of execution time and report results.
"""
from typing import Dict
from executor.run_original import run_original
from executor.run_optimized import run_optimized


def benchmark(adj_original: Dict[str, list], adj_optimized: Dict[str, list], runs: int = 10):
    """Run multiple inferences and compare average timings."""
    import statistics

    orig_times = []
    opt_times = []
    for _ in range(runs):
        _, t1 = run_original()
        _, t2 = run_optimized(adj_optimized)
        orig_times.append(t1)
        opt_times.append(t2)
    avg_orig = sum(orig_times) / len(orig_times)
    avg_opt = sum(opt_times) / len(opt_times)
    std_orig = statistics.stdev(orig_times) if len(orig_times) > 1 else 0.0
    std_opt = statistics.stdev(opt_times) if len(opt_times) > 1 else 0.0
    boost = (avg_orig - avg_opt) / avg_orig * 100.0 if avg_orig != 0 else 0.0

    print("-----------------------------------------")
    print(f"Layers Before: {len(adj_original)}")
    print(f"Layers After: {len(adj_optimized)}")
    print(f"Avg latency (original): {avg_orig:.2f} ms (std: {std_orig:.2f} ms)")
    print(f"Avg latency (optimized): {avg_opt:.2f} ms (std: {std_opt:.2f} ms)")
    print(f"Boost (positive means optimized is faster): {boost:.2f} %")
    print("-----------------------------------------")
    return {
        "layers_before": len(adj_original),
        "layers_after": len(adj_optimized),
        "speed_original": avg_orig,
        "speed_original_std": std_orig,
        "speed_optimized": avg_opt,
        "speed_optimized_std": std_opt,
        "boost": boost,
    }


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    from optimizer.optimize_all import optimize_graph
    opt = optimize_graph(g)
    benchmark(g, opt, runs=3)