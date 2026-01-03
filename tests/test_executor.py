from executor.run_original import run_original
from executor.run_optimized import run_optimized


def test_executors_run():
    out1, t1 = run_original(batch_size=2)
    adj = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": []}
    out2, t2 = run_optimized(adj, batch_size=2)
    assert out1.shape[0] == 2
    assert out2.shape[0] == 2
    assert t1 >= 0
    assert t2 >= 0
