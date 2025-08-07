import torch

class Inferencer:
    def __init__(self, model, inference_loader, device="cpu"):
        self.model = model
        self.inference_loader = inference_loader
        self.device = device

    def infer(self):
        print("🔮 Inferencer: Making predictions...")
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in self.inference_loader:
                inputs = batch.to(self.device)
                outputs = self.model(inputs)

                if outputs.shape[-1] > 1:
                    _, predicted = torch.max(outputs, dim=1)
                else:
                    predicted = torch.round(torch.sigmoid(outputs))

                predictions.extend(predicted.cpu().numpy())

        return predictions
