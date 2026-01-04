"""generate_architecture_variants.py

Generates 5 DFA variants representing the SAME Neural Network architecture at different levels of abstraction/optimization.
This tells the story of the "Compilation" process visually.

1. DFA 1: The Full Original Model (All Layers)
2. DFA 2: The Model with Explicit Dead Code (Before Cleaning)
3. DFA 3: The Optimized Model (Fused Conv+ReLU)
4. DFA 4: The Highly Optimized Model (Fused Everything)
5. DFA 5: The Abstract View (High Level Blocks)
"""
from visualization.draw_dfa import draw_dfa

def generate_dfas():
    # ---------------------------------------------------------
    # DFA 1: The Standard Original Model
    # ---------------------------------------------------------
    s1 = {"Input", "Conv1", "ReLU1", "Pool1", "Conv2", "ReLU2", "Pool2", "Flatten", "FC1", "ReLU3", "FC2", "Output"}
    t1 = {
        "Input": {"Tensor": "Conv1"},
        "Conv1": {"Tensor": "ReLU1"},
        "ReLU1": {"Tensor": "Pool1"},
        "Pool1": {"Tensor": "Conv2"},
        "Conv2": {"Tensor": "ReLU2"},
        "ReLU2": {"Tensor": "Pool2"},
        "Pool2": {"Tensor": "Flatten"},
        "Flatten": {"Tensor": "FC1"},
        "FC1": {"Tensor": "ReLU3"},
        "ReLU3": {"Tensor": "FC2"},
        "FC2": {"Tensor": "Output"},
    }
    draw_dfa(s1, t1, "Input", {"Output"}, "visualizations/dfa1", "DFA 1: Original Architecture")


    # ---------------------------------------------------------
    # DFA 2: Variation Architecture (Fully Connected MLP)
    # ---------------------------------------------------------
    # A distinct valid architecture. Instead of Convolutional (DFA 1),
    # this is a pure Feed-Forward Network (MLP), showing the compiler handles diff structures.
    s2 = {"Input", "Flatten", "Dense_512", "ReLU_1", "Dropout", "Dense_256", "ReLU_2", "Dense_10", "Softmax", "Output"}
    t2 = {
        "Input": {"Tensor": "Flatten"},
        "Flatten": {"Tensor": "Dense_512"},
        "Dense_512": {"Tensor": "ReLU_1"},
        "ReLU_1": {"Tensor": "Dropout"},
        "Dropout": {"Tensor": "Dense_256"},
        "Dense_256": {"Tensor": "ReLU_2"},
        "ReLU_2": {"Tensor": "Dense_10"},
        "Dense_10": {"Tensor": "Softmax"},
        "Softmax": {"Tensor": "Output"}
    }
    
    draw_dfa(s2, t2, "Input", {"Output"}, "visualizations/dfa2", "DFA 2: MLP Variation (Feed-Forward)")


    # ---------------------------------------------------------
    # DFA 3: Partially Optimized (Conv Fused)
    # ---------------------------------------------------------
    # "Conv1 -> ReLU1" becomes "FusedConv1"
    # "Conv2 -> ReLU2" becomes "FusedConv2"
    s3 = {"Input", "FusedConv1", "Pool1", "FusedConv2", "Pool2", "Flatten", "FC1", "ReLU3", "FC2", "Output"}
    t3 = {
        "Input": {"Tensor": "FusedConv1"},
        "FusedConv1": {"Tensor": "Pool1"},
        "Pool1": {"Tensor": "FusedConv2"},
        "FusedConv2": {"Tensor": "Pool2"},
        "Pool2": {"Tensor": "Flatten"},
        "Flatten": {"Tensor": "FC1"},
        "FC1": {"Tensor": "ReLU3"},
        "ReLU3": {"Tensor": "FC2"},
        "FC2": {"Tensor": "Output"},
    }
    draw_dfa(s3, t3, "Input", {"Output"}, "visualizations/dfa3", "DFA 3: Fused Convolutions")


    # ---------------------------------------------------------
    # DFA 4: Fully Optimized (All Fused)
    # ---------------------------------------------------------
    # Also fuse FC1 -> ReLU3 -> FusedFC
    s4 = {"Input", "FusedConv1", "Pool1", "FusedConv2", "Pool2", "Flatten", "FusedFC1", "FC2", "Output"}
    t4 = {
        "Input": {"Tensor": "FusedConv1"},
        "FusedConv1": {"Tensor": "Pool1"},
        "Pool1": {"Tensor": "FusedConv2"},
        "FusedConv2": {"Tensor": "Pool2"},
        "Pool2": {"Tensor": "Flatten"},
        "Flatten": {"Tensor": "FusedFC1"},
        "FusedFC1": {"Tensor": "FC2"},
        "FC2": {"Tensor": "Output"},
    }
    draw_dfa(s4, t4, "Input", {"Output"}, "visualizations/dfa4", "DFA 4: Fully Fused Architecture")


    # ---------------------------------------------------------
    # DFA 5: The Project Workflow (The Compiler "Meta" DFA)
    # ---------------------------------------------------------
    # This DFA represents what the PROJECT does, not the neural network.
    # It explains the Compiler Pipeline itself as a state machine.
    # State 1: We receive a Model (Source)
    # State 2: We Extract the Graph (Parsing)
    # State 3: We Build a DFA (Theory Analysis)
    # State 4: We Generate IR (Intermediate Code)
    # State 5: We Optimize (Fusion/DeadCode)
    # State 6: We Compile & Run (Final Result)
    
    s5 = {"Input", "ConvBlock_1", "ConvBlock_2", "DenseBlock", "Output"}
    t5 = {
        "Input": {"Tensor": "ConvBlock_1"},
        "ConvBlock_1": {"FeatureMap": "ConvBlock_2"},
        "ConvBlock_2": {"FeatureMap": "DenseBlock"},
        "DenseBlock": {"Logits": "Output"}
    }
    draw_dfa(s5, t5, "Input", {"Output"}, "visualizations/dfa5", "DFA 5: Block-Level Architecture (Grouped Layers)")
    
    print("Generated 5 architecture variant DFAs in visualizations/")


if __name__ == "__main__":
    generate_dfas()
