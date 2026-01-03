"""Run a reproducible demo: optionally train with a fixed seed, then run the pipeline to generate report.md and visualizations."""
from model.train_mnist import train
from demo_output import run_demo


def main(train_first: bool = True, seed: int = 42):
    if train_first:
        print("Training model with fixed seed to create reproducible weights (quick run)...")
        train(save_path="model/mnist_cnn.pth", epochs=1, batch_size=32, seed=seed, evaluate=True)
    run_demo()


if __name__ == "__main__":
    main()
