from vio_v3.trainer.validator import Validator
from vio_v3.model.loader import load_model
from vio_v3.model.wrapper import VIOWrapper

import torch
from torch.utils.data import DataLoader

def main():
    print("🔍 Evaluating VIOv3 model...")
    
    core_model = load_model()
    model = VIOWrapper(core_model).to("cuda" if torch.cuda.is_available() else "cpu")

    # NOTE: Replace with actual validation dataset
    val_loader = DataLoader([])

    validator = Validator(model=model, val_loader=val_loader)
    metrics = validator.evaluate()

    print("📊 Evaluation Results:")
    print(metrics)

if __name__ == "__main__":
    main()
