import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()  # Replace with your task-specific loss

    def train(self, epochs=10, lr=1e-4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            print(f"\n🌀 Epoch {epoch + 1}/{epochs}")
            self.model.train()
            running_loss = 0.0

            loop = tqdm(self.train_loader, desc="🔁 Training", leave=False)
            for batch in loop:
                inputs, targets = batch  # Replace with actual batch structure
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(self.train_loader)
            print(f"📉 Avg Loss: {avg_loss:.4f}")

            if self.val_loader:
                self.validate()

    def validate(self):
        print("🔎 Running validation...")
        self.model.eval()
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch  # Replace accordingly
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)

        acc = total_correct / total_samples if total_samples > 0 else 0
        print(f"✅ Validation Accuracy: {acc:.2%}")
