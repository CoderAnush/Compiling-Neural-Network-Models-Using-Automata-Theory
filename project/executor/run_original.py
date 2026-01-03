"""run_original.py

Execute the original PyTorch model for inference on sample input and return outputs + timing.
"""
from typing import Tuple
import time
import torch
from model.train_mnist import get_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def run_original(batch_size: int = 16, device: str = None, use_real_data: bool = False) -> Tuple[torch.Tensor, float]:
    """Run the unmodified model on input and return (outputs, elapsed_ms).

    Args:
        batch_size: number of samples to run in a single forward
        device: device string (e.g., 'cpu' or 'cuda')
        use_real_data: if True, use real MNIST test samples; otherwise use random tensors
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev)
    model = get_model(pretrained=False, device=dev)
    model.to(dev)
    model.eval()

    if use_real_data:
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
    out, t = run_original(use_real_data=False)
    print(f"Original run time: {t:.2f} ms; output shape: {out.shape}")