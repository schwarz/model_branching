import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Tensor of shape [n, m] -> Tensor of shape [m, n, 1]
def shape(tensor: torch.Tensor):
    t = tensor.permute(1, 0)
    return t.view(t.size(0), t.size(1), 1)


# https://discuss.pytorch.org/t/rmsle-loss-function/67281
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
