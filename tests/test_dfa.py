from automata.dfa_builder import build_dfa_from_adjacency, reachability


def test_dfa_and_reachability():
    adj = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": []}
    dfa = build_dfa_from_adjacency(adj)
    reachable, unreachable = reachability(dfa)
    assert dfa.start in reachable
    assert isinstance(unreachable, set)
