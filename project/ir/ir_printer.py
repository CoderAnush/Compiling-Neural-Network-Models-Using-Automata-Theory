"""ir_printer.py

Pretty print IR lists produced by `generate_ir`.
"""
from typing import List, Tuple


def pretty_print_ir(ir: List[str]) -> str:
    """Return a human-readable multi-line representation of IR lines."""
    return "\n".join(ir)


def pretty_print_structured(ir: List[Tuple[str, str, List[str]]]) -> str:
    lines = [f"{dest} = {op}({', '.join(args)})" for dest, op, args in ir]
    return "\n".join(lines)


if __name__ == "__main__":
    sample = ["%1 = Input()", "%2 = Conv(%1)", "%3 = ReLU(%2)"]
    print(pretty_print_ir(sample))
