# Neural Network Compiler — Complete Guide (Beginner to Advanced)

:wave: **Welcome!** This project is an educational demo that bridges the gap between **Deep Learning** (AI) and **Computer Science Theory**.
It shows how we can treat a Neural Network not as a "black box" of math, but as a **program** that can be analyzed, debugged, and optimized using standard compiler rules.

---

## What is this project? (The Simple Explanation)

Imagine you have a factory assembly line (the Neural Network). Raw materials (Data) go in, travel through various machines (Layers), and a finished product (Prediction) comes out.

Sometimes, this assembly line is inefficient:
1.  **Dead Ends:** There might be conveyor belts that lead nowhere. Materials sent there are wasted.
2.  **Redundant Steps:** You might have a machine that paints the product red, followed immediately by a machine that dries the paint. It would be faster to have one machine do both at once.

This project builds a **Compiler** for that factory. It looks at the blueprints, finds these inefficiencies using strict mathematical rules (Automata Theory), and rebuilds a newer, faster factory.

### The 6-Step "Compiler Pipeline" (Detailed Execution)

This project strictly follows the lifecycle of a real compiler. Here is exactly what happens when you run the demo:

#### 1. The Input (The Model)
*   **Execution:** The system first runs `project/model/train_mnist.py`.
*   **What happens:** It creates a standard Convolutional Neural Network (CNN). Think of this as a chain of 7 specialized workers (Layers), such as `Conv` (detects patterns) and `ReLU` (filters negative values).
*   **State:** At this point, the model is just a Python object. The computer treats it as generic code.

#### 2. Graph Extraction (The "Parsing")
*   **Execution:** The script `project/graph/extract_graph.py` inspects the trained model.
*   **What happens:** It "reads" the Python code and converts it into a **Computational Graph** (Node-Edge structure). It produces a JSON file (`graph_original.json`) representing the raw blueprint of the network.
*   **Why:** We cannot optimize Python code easily, but we can easily optimize a graph data structure.

#### 3. Automata Analysis (The "Theory")
*   **Execution:** `project/automata/dfa_builder.py` converts the graph into a **DFA (Deterministic Finite Automaton)**.
*   **The Math:** It defines the network as a machine where:
    *   **States ($Q$):** Each layer of the neural network.
    *   **Transitions ($\delta$):** The data flow between layers.
*   **The Algorithm:** It runs a **Reachability Analysis** (using Depth-First Search). It starts at the `Input` and traces every valid path. Any state that is not visited is mathematically proven to be **Unreachable** (Dead Code).

#### 4. Intermediate Representation (IR)
*   **Execution:** `project/ir/generate_ir.py` converts the graph into text.
*   **What happens:** Compilers work best with linear instructions, not graphs. The system generates a "recipe" that looks like Assembly Language:
    ```text
    %1 = Conv(%0)
    %2 = ReLU(%1)
    ```
*   **Why:** This linear format makes it easier to spot patterns (like two instructions that can be merged).

#### 5. Optimization (The "Cleanup")
*   **Execution:** `project/optimizer/optimize_all.py` runs two major "passes" over the IR:
    1.  **Dead Node Elimination:** It physically identifies and deletes the nodes that Step 3 proved were unreachable.
    2.  **Operator Fusion:** It searches for inefficient patterns. For example, a `Convolution` followed immediately by a `ReLU` is merged into a single `FusedConvReLU` node.
*   **Result:** A new, streamlined graph (`graph_optimized.json`) with fewer nodes and faster operations.

#### 6. Code Generation & Benchmark
*   **Execution:** `project/demo_output.py` takes the new graph and "compiles" it back into runnable code (`run_optimized.py`).
*   **Verification:** It runs a race (Benchmark) between the Original Model and the Optimized Model on the MNIST test data. It generates a report comparing their speed and structure.

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
