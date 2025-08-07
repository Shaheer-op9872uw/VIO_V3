from vio_v3.trainer.trainer import Trainer
from vio_v3.model.loader import load_model
from vio_v3.model.wrapper import VIOWrapper
from vio_v3.utils.metrics import compute_metrics

import torch
from torch.utils.data import DataLoader
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train VIOv3 model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = get_args()

    print("🔧 Loading model...")
    core_model = load_model()
    model = VIOWrapper(core_model).to(args.device)

    print("📦 Loading data...")
    # NOTE: Replace with actual dataset & dataloader
    train_loader = DataLoader([])  # stub
    val_loader = DataLoader([])

    print("🚀 Starting training loop...")
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=args.device)

    trainer.train(epochs=args.epochs, lr=args.lr)

    print("✅ Training complete.")
    print("📊 Final metrics:")
    compute_metrics()  # placeholder

if __name__ == "__main__":
    main()
