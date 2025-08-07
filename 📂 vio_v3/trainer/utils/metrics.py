import torch
import torch.nn.functional as F

class Metrics:
    def __init__(self, task="classification"):
        self.task = task
        self.reset()

    def reset(self):
        self.preds = []
        self.targets = []

    def update(self, outputs, targets):
        if self.task == "classification":
            preds = torch.argmax(outputs, dim=1)
        elif self.task == "regression":
            preds = outputs.squeeze()
        else:
            raise ValueError("Unsupported task type")

        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        if self.task == "classification":
            accuracy = (preds == targets).float().mean().item()
            return {"accuracy": accuracy}
        elif self.task == "regression":
            mse = F.mse_loss(preds, targets).item()
            mae = F.l1_loss(preds, targets).item()
            return {"mse": mse, "mae": mae}
        else:
            return {}

    def __str__(self):
        metrics = self.compute()
        return " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
