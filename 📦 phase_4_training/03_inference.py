from vio_v3.trainer.inferencer import Inferencer
from vio_v3.model.loader import load_model
from vio_v3.model.wrapper import VIOWrapper

import torch
from torch.utils.data import DataLoader

def main():
    print("🧠 Inference on new data using VIOv3...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    core_model = load_model()
    model = VIOWrapper(core_model).to(device)

    # NOTE: Replace with actual inference dataset
    inference_loader = DataLoader([])

    inferencer = Inferencer(model=model, inference_loader=inference_loader)
    predictions = inferencer.infer()

    print("🔮 Predictions:")
    for idx, output in enumerate(predictions):
        print(f"[{idx}] {output}")

if __name__ == "__main__":
    main()
