from benchmark.benchmark_compare import benchmark


def test_benchmark_structure():
    adj = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": []}
    opt = {"Input": ["FusedConvReLU"], "FusedConvReLU": ["ReLU"], "ReLU": []}
    result = benchmark(adj, opt, runs=1)
    # Should include expected keys
    assert "layers_before" in result and "layers_after" in result
    assert "speed_original" in result and "speed_optimized" in result
    assert "boost" in result
