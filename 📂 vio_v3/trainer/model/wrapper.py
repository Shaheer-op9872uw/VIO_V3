import torch.nn as nn

class VIOWrapper(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.config = config

        if config.get("task") == "classification":
            self.head = nn.Linear(config["hidden_dim"], config["num_classes"])
        elif config.get("task") == "regression":
            self.head = nn.Linear(config["hidden_dim"], 1)
        else:
            raise ValueError("Unsupported task type")

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
