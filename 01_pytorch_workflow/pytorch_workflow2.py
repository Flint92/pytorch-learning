import torch

from torch import nn
from utils.device import get_device

class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
    # define the forward computation (input data x -> output data)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

if __name__ == '__main__':
    torch.manual_seed(42)
    model_1 = LinearRegressionModelV2()
    print(model_1.state_dict())

    print(next(model_1.parameters()).device)
    model_1.to(get_device())
    print(next(model_1.parameters()).device)

