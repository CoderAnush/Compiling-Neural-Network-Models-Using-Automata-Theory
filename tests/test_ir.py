from ir.generate_ir import graph_to_ir, ir_as_list


def test_ir_consistency():
    adj = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": []}
    lines = graph_to_ir(adj)
    struct = ir_as_list(adj)
    assert len(lines) == len(struct)
    assert all(isinstance(s, str) for s in lines)
