# Theory of Automata: 5 Representations of the Neural Network

This document provides a comprehensive theoretical analysis of the 5 variations of the Neural Network Compiler project. Each variant is modeled as a **Deterministic Finite Automaton (DFA)**.

For each DFA, we provide:
1.  **Transition Table**: The logic gates of the state machine.
2.  **Regular Grammar**: The derivation rules (Type-3 Grammar).
3.  **Regular Expression**: The pattern string accepted by the machine.
4.  **Detailed Explanation**: Formal analysis of the structure.

---

# DFA 1: The Original Architecture (CNN)
**Description:** This represents the full, unoptimized Convolutional Neural Network used as the input to the compiler.

### 1. Transition Table
| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) |
| :--- | :---: | :--- |
| **Input** | Tensor | Conv1 |
| **Conv1** | Tensor | ReLU1 |
| **ReLU1** | Tensor | Pool1 |
| **Pool1** | Tensor | Conv2 |
| **Conv2** | Tensor | ReLU2 |
| **ReLU2** | Tensor | Pool2 |
| **Pool2** | Tensor | Flatten |
| **Flatten** | Tensor | FC1 |
| **FC1** | Tensor | ReLU3 |
| **ReLU3** | Tensor | FC2 |
| **FC2** | Tensor | Output |
| **Output** | Tensor | $\emptyset$ (Halt) |

### 2. Regular Grammar
**Variables:** $S, A, B, C, D, E, F, G, H, I, J$
**Terminal:** $t$ (Tensor)

*   $S \to tA$ (Input $\to$ Conv1)
*   $A \to tB$ (Conv1 $\to$ ReLU1)
*   $B \to tC$ (ReLU1 $\to$ Pool1)
*   $C \to tD$ (Pool1 $\to$ Conv2)
*   $D \to tE$ (Conv2 $\to$ ReLU2)
*   $E \to tF$ (ReLU2 $\to$ Pool2)
*   $F \to tG$ (Pool2 $\to$ Flatten)
*   $G \to tH$ (Flatten $\to$ FC1)
*   $H \to tI$ (FC1 $\to$ ReLU3)
*   $I \to tJ$ (ReLU3 $\to$ FC2)
*   $J \to t\text{Output}$
*   $\text{Output} \to \epsilon$

### 3. Regular Expression
$$ L(M_1) = \text{Input} \cdot \text{Conv1} \cdot \text{ReLU1} \cdot \text{Pool1} \cdot \text{Conv2} \cdot \text{ReLU2} \cdot \text{Pool2} \cdot \text{Flatten} \cdot \text{FC1} \cdot \text{ReLU3} \cdot \text{FC2} \cdot \text{Output} $$

### 4. Explanation
This DFA represents the **State Space** of the raw source code. It is a strictly linear chain with 12 states. The "Alphabet" consists of Tensors passing between these layers. If any transition is missing or reordered (e.g., `Pool` before `ReLU`), the string is rejected, proving the model invalid.

---

# DFA 2: The MLP Variation (Feed-Forward)
**Description:** A completely different valid architecture (Multi-Layer Perceptron) to demonstrate the compiler's flexibility.

### 1. Transition Table
| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) |
| :--- | :---: | :--- |
| **Input** | Tensor | Flatten |
| **Flatten** | Tensor | Dense_512 |
| **Dense_512** | Tensor | ReLU_1 |
| **ReLU_1** | Tensor | Dropout |
| **Dropout** | Tensor | Dense_256 |
| **Dense_256** | Tensor | ReLU_2 |
| **ReLU_2** | Tensor | Dense_10 |
| **Dense_10** | Tensor | Softmax |
| **Softmax** | Tensor | Output |

### 2. Regular Grammar
**Variables:** $S, A, B, C, D, E, F, G, H$
**Terminal:** $t$ (Tensor)

*   $S \to tA$ (Input $\to$ Flatten)
*   $A \to tB$ (Flatten $\to$ Dense_512)
*   $B \to tC$ (Dense_512 $\to$ ReLU_1)
*   $C \to tD$ (ReLU_1 $\to$ Dropout)
*   $D \to tE$ (Dropout $\to$ Dense_256)
*   $E \to tF$ (Dense_256 $\to$ ReLU_2)
*   $F \to tG$ (ReLU_2 $\to$ Dense_10)
*   $G \to tH$ (Dense_10 $\to$ Softmax)
*   $H \to t\text{Output}$
*   $\text{Output} \to \epsilon$

### 3. Regular Expression
$$ L(M_2) = \text{Input} \cdot \text{Flatten} \cdot \text{Dense}_{512} \cdot \text{ReLU} \cdot \text{Dropout} \cdot \text{Dense}_{256} \cdot \text{ReLU} \cdot \text{Dense}_{10} \cdot \text{Softmax} \cdot \text{Output} $$

### 4. Explanation
This variation validates a Dense network. Note how the grammar structure ($S \to tA \to tB$) remains identical to DFA 1, but the **Symbols** (Layer types) differ. This proves that the underlying theoretical model (Right-Linear Grammar) applies universally to any sequential Neural Network.

---

# DFA 3: Partially Optimized (Fused Convolution)
**Description:** The result of the first optimization pass. `Conv` + `ReLU` states have been merged.

### 1. Transition Table
| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) |
| :--- | :---: | :--- |
| **Input** | Tensor | FusedConv1 |
| **FusedConv1** | Tensor | Pool1 |
| **Pool1** | Tensor | FusedConv2 |
| **FusedConv2** | Tensor | Pool2 |
| **Pool2** | Tensor | Flatten |
| **Flatten** | Tensor | FC1 |
| **FC1** | Tensor | ReLU3 |
| **ReLU3** | Tensor | FC2 |
| **FC2** | Tensor | Output |

### 2. Regular Grammar
*   $S \to tA$ (Input $\to$ FusedConv1)
*   $A \to tB$ (FusedConv1 $\to$ Pool1)
*   $B \to tC$ (Pool1 $\to$ FusedConv2)
*   $C \to tD$ (FusedConv2 $\to$ Pool2)
*   $D \to tE$ (Pool2 $\to$ Flatten)
*   $E \to tF$ (Flatten $\to$ FC1)
*   $F \to tG$ (FC1 $\to$ ReLU3)
*   $G \to tH$ (ReLU3 $\to$ FC2)
*   $H \to t\text{Output}$

### 3. Regular Expression
$$ L(M_3) = \text{Input} \cdot \textbf{FusedConv1} \cdot \text{Pool1} \cdot \textbf{FusedConv2} \cdot \text{Pool2} \cdot \text{Flatten} \cdot \text{FC1} \cdot \text{ReLU3} \cdot \text{FC2} \cdot \text{Output} $$

### 4. Explanation
The graph has shrunk. We removed 2 states (`Conv1`, `ReLU1`) and replaced them with 1 state (`FusedConv1`).
The grammar derivation is **shorter** (fewer steps to reach $\epsilon$). This mathematically guarantees that the program execution path is shorter, leading to potential performance gains (less overhead).

---

# DFA 4: Fully Optimized (Fused Dense)
**Description:** The final optimized form. Even the Linear layers have been fused (`FC` + `ReLU`).

### 1. Transition Table
| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) |
| :--- | :---: | :--- |
| **Input** | Tensor | FusedConv1 |
| **FusedConv1** | Tensor | Pool1 |
| **Pool1** | Tensor | FusedConv2 |
| **FusedConv2** | Tensor | Pool2 |
| **Pool2** | Tensor | Flatten |
| **Flatten** | Tensor | FusedFC1 |
| **FusedFC1** | Tensor | FC2 |
| **FC2** | Tensor | Output |

### 2. Regular Grammar
*   $S \to tA$ (Input $\to$ FusedConv1)
*   $A \to tB$ (FusedConv1 $\to$ Pool1)
*   $B \to tC$ (Pool1 $\to$ FusedConv2)
*   $C \to tD$ (FusedConv2 $\to$ Pool2)
*   $D \to tE$ (Pool2 $\to$ Flatten)
*   $E \to tF$ (Flatten $\to$ FusedFC1)
*   $F \to tG$ (FusedFC1 $\to$ FC2)
*   $G \to t\text{Output}$

### 3. Regular Expression
$$ L(M_4) = \text{Input} \cdot \textbf{FusedConv1} \cdot \text{Pool1} \cdot \textbf{FusedConv2} \cdot \text{Pool2} \cdot \text{Flatten} \cdot \textbf{FusedFC1} \cdot \text{FC2} \cdot \text{Output} $$

### 4. Explanation
This is the **Minimal DFA** for the functional logic of the network.
*   **Original Nodes:** 12
*   **Optimized Nodes:** 8
*   **Reduction:** 33% reduction in state transitions.
By minimizing the states in the DFA, we minimize the memory read/write operations in the actual computer hardware.

---

# DFA 5: Block-Level Abstraction
**Description:** A high-level architectural view, grouping layers into "Blocks".

### 1. Transition Table
| Current State ($q$) | Input Symbol ($\sigma$) | Next State ($\delta(q, \sigma)$) |
| :--- | :---: | :--- |
| **Input** | Tensor | ConvBlock_1 |
| **ConvBlock_1** | FeatureMap | ConvBlock_2 |
| **ConvBlock_2** | FeatureMap | DenseBlock |
| **DenseBlock** | Logits | Output |

### 2. Regular Grammar
**Variables:** $S, A, B, C$
**Terminals:** $t$ (Tensor), $f$ (FeatureMap), $l$ (Logits)

*   $S \to tA$ (Input $\to$ ConvBlock_1)
*   $A \to fB$ (ConvBlock_1 $\to$ ConvBlock_2)
*   $B \to fC$ (ConvBlock_2 $\to$ DenseBlock)
*   $C \to l\text{Output}$
*   $\text{Output} \to \epsilon$

### 3. Regular Expression
$$ L(M_5) = \text{Input} \cdot \text{ConvBlock}_1 \cdot \text{ConvBlock}_2 \cdot \text{DenseBlock} \cdot \text{Output} $$

### 4. Explanation
This represents the **Meta-Grammar** of the architecture. Instead of analyzing individual operations (micro-states), we analyze the flow between modules (macro-states). This is equivalent to "Function Call" graphs in compiler design, whereas DFA 1 is equivalent to "Instruction Flow".
