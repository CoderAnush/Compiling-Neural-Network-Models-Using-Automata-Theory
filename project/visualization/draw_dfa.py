"""draw_dfa.py

Render DFA as PNG diagrams using graphviz (via pydot or graphviz).
"""
from typing import Dict, Set
import graphviz


def draw_dfa(states: Set[str], transitions: Dict[str, Dict[str, str]], start: str, accept: Set[str], path: str = "dfa.png", title: str = "DFA"):
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="LR")
    g.attr(label=title)
    for s in states:
        shape = "doublecircle" if s in accept else "circle"
        g.node(s, shape=shape)
    # add invisible start arrow
    g.node("", shape="none")
    g.edge("", start)
    for s, trans in transitions.items():
        for sym, nxt in trans.items():
            g.edge(s, nxt, label=sym)
    g.render(path, cleanup=True)


if __name__ == "__main__":
    states = {"Input", "Conv", "ReLU", "Pool", "Dense", "Softmax"}
    trans = {"Input": {"Tensor": "Conv"}, "Conv": {"Tensor": "ReLU"}, "ReLU": {"Tensor": "Pool"}, "Pool": {"Tensor": "Dense"}, "Dense": {"Tensor": "Softmax"}, "Softmax": {"Tensor": "Softmax"}}
    draw_dfa(states, trans, "Input", {"Softmax"}, "dfa1", "Demo DFA")
    print("Saved dfa1.png")