import os
import torch
from torch.utils.data import TensorDataset
from model.train_mnist import MNISTCNN, train, get_model


def test_mnist_forward():
    model = MNISTCNN()
    x = torch.randn(1, 1, 28, 28)
    out = model(x)
    assert out.shape == (1, 10)
    assert torch.isfinite(out).all()


def test_train_with_synthetic_dataset(tmp_path):
    # Create a tiny synthetic dataset (4 samples)
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    ds = TensorDataset(x, y)

    save_path = tmp_path / "mnist_test.pth"
    # Train for 1 epoch on the synthetic dataset
    path = train(save_path=str(save_path), epochs=1, batch_size=2, train_ds=ds)
    assert os.path.exists(path)

    # Ensure get_model can load the saved weights
    m = get_model(pretrained=True, path=str(save_path))
    out = m(torch.randn(1, 1, 28, 28))
    assert out.shape == (1, 10)
