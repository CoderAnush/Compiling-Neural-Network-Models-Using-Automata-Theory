"""dfa_builder.py

Convert a computational graph (adjacency list) into a DFA-like representation.

We define:
    Q = set(layers)
    Σ = {Tensor} (single symbol representing data flow)
    δ: mapping from state -> dict(symbol -> next_state)
    q0 = 'Input'
    F = {'Output'} or layers with no outgoing edges

Provides reachability analysis to find dead/unreachable states.
"""
from typing import Dict, List, Set, Tuple


class DFA:
    """A simple DFA representation for our computational graph."""

    def __init__(self, states: Set[str], alphabet: Set[str], transitions: Dict[str, Dict[str, str]], start: str, accept: Set[str]):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start = start
        self.accept = accept

    def __repr__(self):
        return f"DFA(states={self.states}, start={self.start}, accept={self.accept})"


def build_dfa_from_adjacency(adj: Dict[str, List[str]]) -> DFA:
    """Build a DFA from adjacency list where each edge is labeled 'Tensor'.

    Args:
        adj: adjacency dict mapping node->list of next nodes

    Returns:
        DFA instance
    """
    states = set(adj.keys())
    # include nodes that appear as targets but not as keys
    for targets in adj.values():
        for t in targets:
            states.add(t)

    alphabet = {"Tensor"}
    transitions = {}
    for s, targets in adj.items():
        transitions[s] = {"Tensor": targets[0] if targets else s}  # deterministic pick first or self-loop for no target

    start = "Input" if "Input" in states else next(iter(states))
    # accept states are nodes with no outgoing edges or 'Output'
    accept = {s for s, t in adj.items() if len(t) == 0}
    if "Output" in states:
        accept.add("Output")
    return DFA(states, alphabet, transitions, start, accept)


def reachability(dfa: DFA) -> Tuple[Set[str], Set[str]]:
    """Return (reachable_states, unreachable_states) from start state."""
    visited = set()
    stack = [dfa.start]
    while stack:
        s = stack.pop()
        if s in visited:
            continue
        visited.add(s)
        if s in dfa.transitions:
            for sym, nxt in dfa.transitions[s].items():
                if nxt not in visited:
                    stack.append(nxt)
    unreachable = set(dfa.states) - visited
    return visited, unreachable


if __name__ == "__main__":
    demo = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    dfa = build_dfa_from_adjacency(demo)
    print(dfa)
    r, u = reachability(dfa)
    print("Reachable:", r)
    print("Unreachable:", u)
