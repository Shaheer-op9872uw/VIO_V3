import torch

class Validator:
    def __init__(self, model, val_loader, device="cpu"):
        self.model = model
        self.val_loader = val_loader
        self.device = device

    def validate(self):
        print("🔍 Validator: Running evaluation...")
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        criterion = torch.nn.CrossEntropyLoss()  # Change if needed

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                _, preds = torch.max(outputs, dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        print(f"📈 Validator Results → Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
