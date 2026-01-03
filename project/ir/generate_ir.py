"""generate_ir.py

Generate a simple linear IR from an adjacency list graph. The IR uses SSA-like numbering:

%1 = Conv(Input)
%2 = ReLU(%1)
%3 = MaxPool(%2)
... etc

Functions:
- graph_to_ir(graph): returns list of IR strings
- ir_as_list(graph): returns structured IR entries as tuples (dest, op, args)
"""
from typing import Dict, List, Tuple


def graph_to_ir(adj: Dict[str, List[str]]) -> List[str]:
    """Convert adjacency into textual IR list. Assumes linear chain graph.

    Returns list of strings representing IR instructions.
    """
    mapping = {}
    ir = []
    counter = 1

    # build linear order by walking from Input
    curr = "Input"
    while True:
        # handle Input specially
        if curr not in mapping:
            mapping[curr] = f"%{counter}"
            counter += 1
            ir.append(f"{mapping[curr]} = {curr}()" if curr == "Input" else f"{mapping[curr]} = {curr}({prev_reg})")
        # move to next
        nexts = adj.get(curr, [])
        if not nexts:
            break
        prev_reg = mapping[curr]
        curr = nexts[0]
    return ir


def ir_as_list(adj: Dict[str, List[str]]) -> List[Tuple[str, str, List[str]]]:
    """Return structured IR as tuples (dest, op, args)."""
    mapping = {}
    ir = []
    counter = 1
    curr = "Input"
    prev_reg = None
    while True:
        if curr not in mapping:
            dest = f"%{counter}"
            mapping[curr] = dest
            args = [] if prev_reg is None else [prev_reg]
            ir.append((dest, curr, args))
            counter += 1
        nexts = adj.get(curr, [])
        if not nexts:
            break
        prev_reg = mapping[curr]
        curr = nexts[0]
    return ir


if __name__ == "__main__":
    g = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    print("IR lines:")
    for l in graph_to_ir(g):
        print(l)
    print("Structured IR:")
    for t in ir_as_list(g):
        print(t)
