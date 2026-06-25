import torch
from torch import nn


class SemObjective(nn.Module):
    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        residual = data - data @ weights
        return 0.5 * (residual * residual).mean()


def export_onnx(path: str, order: int, samples: int = 8) -> None:
    model = SemObjective().eval()
    data = torch.zeros(samples, order, dtype=torch.float32)
    weights = torch.zeros(order, order, dtype=torch.float32)
    torch.onnx.export(
        model,
        (data, weights),
        path,
        input_names=["data", "weights"],
        output_names=["objective"],
        opset_version=17,
        dynamo=False,
    )
