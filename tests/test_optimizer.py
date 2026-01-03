from optimizer.optimize_all import optimize_graph


def test_optimize_fusion():
    adj = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": []}
    opt = optimize_graph(adj)
    # Expect fused node name present after optimization
    keys = set(opt.keys())
    assert any("FusedConvReLU" in k for k in keys) or any(
        "FusedConvReLU" in v for v in str(opt.values())
    )


def test_dead_node_elimination():
    # Create a graph where an unused tail exists
    adj = {
        "Input": ["Conv"],
        "Conv": ["ReLU"],
        "ReLU": [],
        "Dead": ["Unused"],
        "Unused": [],
    }
    opt = optimize_graph(adj)
    # Dead nodes should not be present after optimization
    assert "Dead" not in opt and "Unused" not in opt
