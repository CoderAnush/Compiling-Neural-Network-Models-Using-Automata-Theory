"""run_optimized.py

Map optimized graph nodes back to an executable PyTorch model (best-effort mapping) and run inference.
This module supports fused operators like FusedConvReLU by implementing equivalent PyTorch sequence.
"""
from typing import Dict, List, Tuple
import time
import torch
from torch import nn
from model.train_mnist import get_model


class ReshapeDenseLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)


class OptimizedModel(nn.Module):
    """A simple runtime that constructs layers according to an adjacency graph.

    For demo purposes, we'll build a sequential model by mapping node names to layer instances:
    - Conv -> Conv2d(1,8)
    - ReLU -> ReLU
    - MaxPool -> MaxPool2d
    - Flatten -> Flatten
    - Dense -> Linear
    - Softmax -> LogSoftmax
    - FusedConvReLU -> Conv2d + ReLU fused into Sequential
    - ReshapeDense -> Flatten + Linear fused into custom module

    Note: This is a best-effort mapping for demonstration; in production you'd preserve shapes and parameters.
    """

    def __init__(self, adj: Dict[str, List[str]]):
        super().__init__()
        layers = []

        # Build linear order from Input
        curr = "Input"
        visited = set()
        while True:
            if curr in visited:
                break
            visited.add(curr)
            nexts = adj.get(curr, [])
            if not nexts:
                break
            nxt = nexts[0]
            # map nxt to layer with rudimentary shape inference based on name
            # (In a real compiler, we would propagate shapes in the IR)
            if nxt.startswith("Conv"):
                if "features_0" in nxt:
                    layers.append(nn.Conv2d(1, 8, kernel_size=3, padding=1))
                elif "features_3" in nxt:
                    layers.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
                else:
                    layers.append(nn.Conv2d(1, 8, kernel_size=3, padding=1)) # fallback
            
            elif nxt.startswith("ReLU"):
                layers.append(nn.ReLU())
            
            elif nxt.startswith("MaxPool"):
                layers.append(nn.MaxPool2d(2))
            
            elif nxt.startswith("Flatten"):
                layers.append(nn.Flatten())
            
            elif nxt.startswith("Dense"):
                if "classifier_0" in nxt or "classifier" in nxt and "2" not in nxt: 
                    # First dense layer (after flatten)
                    layers.append(nn.Linear(16 * 7 * 7, 64))
                elif "classifier_2" in nxt:
                    # Output dense layer
                    layers.append(nn.Linear(64, 10))
                else:
                    layers.append(nn.Linear(64, 10)) # fallback

            elif nxt.startswith("Softmax"):
                layers.append(nn.LogSoftmax(dim=1))
            
            elif nxt.startswith("FusedConvReLU"):
                if "features_0" in nxt:
                    layers.append(nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU()))
                elif "features_3" in nxt:
                    layers.append(nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU()))
                else:
                    layers.append(nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU()))

            elif nxt.startswith("ReshapeDense"):
                # Usually corresponds to Flatten -> Dense(16*7*7 -> 64)
                # Re-introduced flattening logic since the graph node removed it
                layers.append(ReshapeDenseLayer(16 * 7 * 7, 64))
            
            else:
                # unknown node; insert identity mapping
                layers.append(nn.Identity())
            curr = nxt

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_optimized(adj: Dict[str, List[str]], batch_size: int = 16, device: str = None, use_real_data: bool = False) -> Tuple[torch.Tensor, float]:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev)
    model = OptimizedModel(adj).to(dev)
    model.eval()

    if use_real_data:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        x, _ = next(iter(loader))
        x = x.to(dev)
    else:
        x = torch.randn(batch_size, 1, 28, 28, device=dev)

    with torch.no_grad():
        start = time.time()
        out = model(x)
        elapsed = (time.time() - start) * 1000.0
    return out, elapsed


if __name__ == "__main__":
    demo = {"Input": ["Conv"], "Conv": ["ReLU"], "ReLU": ["MaxPool"], "MaxPool": ["Flatten"], "Flatten": ["Dense"], "Dense": ["Softmax"], "Softmax": []}
    out, t = run_optimized(demo)
    print(f"Optimized run time: {t:.2f} ms; output shape: {out.shape}")