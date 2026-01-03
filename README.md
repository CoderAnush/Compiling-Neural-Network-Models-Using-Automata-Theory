# Neural Network Compiler — Complete Guide (Beginner to Advanced)

:wave: Welcome! This repository is an educational demo that shows how a tiny neural network (a simple MNIST CNN) can be represented as a graph, transformed using basic compiler-style optimizations, and executed again to compare performance and structure.

This guide is written for *everyone* — if you're new to neural networks or compilers, read the "For beginners" sections first; if you're a developer, skip to the "For maintainers and contributors" section.

This README explains the project in plain language and gives step‑by‑step instructions to reproduce results.

---

## What this project does (high level)

- Trains or loads a small convolutional neural network (CNN) that can classify MNIST digits (0–9).
- Converts the trained model into a node graph (each layer becomes a node).
- Builds an automaton (a DFA) from the graph and analyzes which parts are reachable.
- Generates a simple intermediate representation (IR) from the graph (like a program listing of operations).
- Applies a few demonstration optimizations (e.g., fuse a Conv+ReLU into a single fused node, remove dead nodes).
- Maps the optimized graph back to a runnable PyTorch model (best-effort mapping; weights are not preserved by the optimizer in this demo).
- Runs both the original and optimized models, benchmarks them, generates visualizations, and saves a human-readable `report.md`.

Why this is useful: it shows how compiler ideas (graph transforms, IR, optimization passes) can apply to neural networks in a small, easy-to-follow example.

---

## Quick start — reproduce the demo (5 minutes)

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.\.venv\Scripts\activate  # Windows (PowerShell)
```

2) Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

3) Run the full demo which will (a) load the model, (b) extract the graph, (c) optimize it, (d) run both models on the MNIST test set, (e) benchmark and save visualizations and a `report.md`:

```bash
python project/demo_output.py
```

Outputs you should see:
- `report.md` — a human-readable summary of the graphs, IR, and a small benchmark
- `visualizations/graph_before.png` and `visualizations/graph_after.png` — visual graphs
- (Optional) `visualizations/dfa1.*` — DFA diagram (requires `graphviz` library)

Notes:
- The demo downloads the MNIST dataset automatically into `./data` if necessary.
- If your machine does not have a network connection, the demo will fall back to random inputs to finish the pipeline (the numbers will not be meaningful in that case).

---

## Developer tasks and how to run them

- Run unit tests locally:

```bash
pytest -q
```

- Check style locally using `ruff`:

```bash
ruff check .
```

- Run the training script manually (small example):

```bash
python project/model/train_mnist.py --train --epochs 2 --save_path model/mnist_cnn.pth --evaluate
```

Important: the training helper accepts a `--seed` and a `--data_dir` for reproducibility and for testing purposes. The `train()` function also accepts a synthetic `train_ds` dataset so tests can call it without downloading MNIST.

---

## What is included (brief folder guide)

- `project/model` — model implementation and training utilities (`train_mnist.py`).
- `project/graph` — extract and save the computational graph (`extract_graph.py`).
- `project/automata` — build DFA from the graph and compute reachability.
- `project/ir` — generate a simple IR (text-like list of ops) and pretty-print it.
- `project/optimizer` — small optimizer passes (dead node elimination, fusion).
- `project/executor` — small runtimes that run the original and the optimized graph (best-effort mapping).
- `project/benchmark` — runs repeated inference and reports average timings (+ calculated "boost" percent).
- `project/visualization` — helpers to draw graphs and optionally DFA diagrams (Graphviz required).
- `project/demo_output.py` — orchestrates the pipeline and writes `report.md`.
- `tests/` — unit tests to validate the pieces (run with `pytest`).

---

## Benchmark & "Boost" explanation (plain English)

- The benchmark measures the **time it takes to run a single forward pass** (in milliseconds) for both the original and the optimized model.
- "Boost" is computed as the percentage reduction in time: (orig - opt) / orig * 100.
  - A **positive** boost means the optimized model is **faster**.
  - A **negative** boost means the optimized model is **slower**.
- Note: In this small demo the optimized model is a structural rewrite and does not preserve learned weights; results vary and can sometimes be slower—this project is focused on demonstrating the optimization pipeline, not production-quality speedups.

---

## Reproducibility notes

- Use `--seed` when training or call `train(..., seed=42)` to make runs deterministic where possible.
- The project uses pinned dependencies in `requirements.txt` so CI and other machines run the same package versions.

---

## Detailed walkthrough (for beginners)

If you're new to neural networks and compilers, here's a gentle, step-by-step explanation of what this demo does.

1. Train or load a small MNIST neural network (the "model").
   - Think of the model as a tiny factory that turns input images (28x28 pixels) into a guess (a number 0–9).
   - The model contains layers like "Conv" (looks for patterns), "ReLU" (applies a simple rule), "MaxPool" (summarizes), "Flatten" (prepares data), and "Dense" (makes the final decision).

2. Convert the model into a graph.
   - Each layer becomes a node and the connections are edges. This is like drawing a flowchart of the factory.

3. Build an automaton and analyze reachability.
   - We convert the graph to a small state machine (DFA) to check which parts of the model are actually used.
   - If a node is unreachable, it means it's never used and can be removed (dead code).

4. Generate an Intermediate Representation (IR).
   - The IR is a textual, step-by-step listing of the operations (like a recipe).
   - This representation is easier to apply transformations to systematically.

5. Apply simple optimizations.
   - Example: Fuse a Conv + ReLU into a single fused operation to avoid redundant passes over the data.
   - Another example: Remove dead nodes discovered by reachability analysis.

6. Map the optimized graph back to a runnable PyTorch model.
   - For this demo we create a new small PyTorch model that corresponds to the optimized graph.
   - Important: in this educational demo we do **not** preserve learned weights across this rewrite (we note it clearly). Preserving weights requires shape propagation and parameter mapping and is left as future work.

7. Run both original and optimized models and benchmark them.
   - We run a few forward passes, measure average latency (milliseconds), and report the percent "boost" (positive = faster).

8. Create visualizations and a human-readable report.
   - `visualizations/graph_before.png` and `visualizations/graph_after.png` show the graphs.
   - `report.md` summarizes graphs, IR, and benchmark results.

---

## Example output snippet (what to expect in `report.md`)

- Graph (original): a JSON adjacency list showing nodes and connections.
- IR (original): a small textual listing like `%1 = Input(); %2 = Conv(%1); %3 = ReLU(%2); ...`
- Graph (optimized): a JSON adjacency list after optimizations (e.g., `FusedConvReLU` nodes appear).
- Benchmark: JSON with average latencies and a boost percent indicating speed change.

---

## Continuous Integration (CI)

This repository includes a GitHub Actions workflow in `.github/workflows/ci.yml` that runs tests and linters on each push/PR to `main`. This helps catch regressions early and ensures code quality.

---

## I want to contribute — what's next?

- Add more optimization passes (e.g., constant folding, shape propagation).
- Preserve weights across optimizations (more work: propagate parameters, shape inference).
- Add integration tests that verify numerical equivalence between original and optimized models when weights are preserved.

---

If you'd like, I can now:
1. Run the test suite and fix any failing tests, and
2. Implement a reproducible demo script and polish the README text further to include step-by-step screenshots and example outputs.

Reproducible demo

To run a reproducible demo that first trains a small model using a fixed seed and then runs the full pipeline:

```bash
python scripts/run_demo.py
```

---

## Glossary (simple definitions)

- Model: the neural network that takes images and outputs a digit prediction.
- Layer: a step inside the model (Conv, ReLU, MaxPool, Flatten, Dense).
- Graph: nodes and edges describing how data flows between layers.
- DFA (Deterministic Finite Automaton): a formal model that helps detect unreachable nodes.
- IR (Intermediate Representation): a textual list of operations used to run transformations.
- Optimization pass: a single transformation on the graph (fusion, removal, rewrite).

---

## Example `report.md` snippet (what you might see)

```json
{
  "layers_before": 13,
  "layers_after": 10,
  "speed_original": 0.6871223449707031,
  "speed_optimized": 1.0495185852050781,
  "boost": -52.74
}
```

- In this example, `boost` is negative (optimized slower). This can happen in small demos and when weights are not preserved.

---

## Clean-up and housekeeping

I can remove compiled artifacts (`__pycache__`, `*.pyc`) and add a `.gitignore` so the repository stays clean. After cleanup I'll re-run tests and linters to ensure everything still passes.

---

If you'd like, I can now:
1. Remove compiled files and unnecessary artifacts and re-run tests/linter, then
2. Prepare a tidy commit or a PR containing all the cleanup and the improved README.

Which should I do next? (I can start the cleanup now.)
