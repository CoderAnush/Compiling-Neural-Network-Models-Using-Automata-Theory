"""train_mnist.py

Training and model loading utilities for a small MNIST CNN.

This module provides:
- MNISTCNN: simple convolutional network
- train(): train on MNIST dataset and save weights
- get_model(): load pretrained weights quickly for downstream pipeline

Usage:
    python train_mnist.py --train --epochs 2 --save_path model/mnist_cnn.pth

Note: training is optional; by default `get_model()` will attempt to load saved weights if present.
"""

from typing import Optional
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTCNN(nn.Module):
    """A small CNN for MNIST classification.

    Architecture:
        Conv(1->8,3x3) -> ReLU -> MaxPool
        Conv(8->16,3x3) -> ReLU -> MaxPool
        Flatten -> FC(256->64) -> ReLU -> FC(64->10) -> LogSoftmax
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # after two 2x pooling, 28x28 -> 7x7
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train(
    save_path: str = "model/mnist_cnn.pth",
    epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    *,
    seed: Optional[int] = None,
    train_ds: Optional[torch.utils.data.Dataset] = None,
    evaluate: bool = False,
    data_dir: str = "./data",
):
    """Train the MNISTCNN on MNIST and save the weights.

    Args:
        save_path: where to save the model state_dict
        epochs: number of epochs to train
        batch_size: training batch size
        lr: learning rate
        device: torch device, default cuda if available else cpu
    Returns:
        path to saved weights
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # set seeds for reproducibility (optional)
    if seed is not None:
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Allow passing a dataset (useful for tests) or download MNIST
    if train_ds is None:
        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MNISTCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    model.train()
    start = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]  loss={loss.item():.4f}")
        print(f"Epoch {epoch} finished. Avg loss: {total_loss / len(train_loader):.4f}")

    # Optionally evaluate on MNIST test set
    if evaluate:
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total if total > 0 else 0.0
        print(f"Test accuracy: {acc*100:.2f}%")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}. Training time: {time.time() - start:.1f}s")
    return save_path


def get_model(pretrained: bool = True, path: str = "model/mnist_cnn.pth", device: Optional[torch.device] = None) -> MNISTCNN:
    """Return MNISTCNN. If `pretrained` is True, attempt to load weights from `path`.

    If weights are not present and `pretrained` is True, an untrained model is returned.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = MNISTCNN().to(device)
    if pretrained and os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"Loaded pretrained weights from {path}")
    else:
        if pretrained:
            print(f"Pretrained weights not found at {path}. Returning untrained model.")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a small MNIST CNN")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="model/mnist_cnn.pth")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset directory")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on test set after training")
    args = parser.parse_args()

    if args.train:
        train(save_path=args.save_path, epochs=args.epochs, seed=args.seed, evaluate=args.evaluate, data_dir=args.data_dir)
    else:
        print("No action requested. Use --train to train the model or import get_model() from this module.")
