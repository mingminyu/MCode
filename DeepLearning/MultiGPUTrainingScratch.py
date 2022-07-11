import torch
from torch import nn
from typing import List, Any
from torch.nn import functional as F


__doc__ = """单机多GPU训练"""


def leNet(x, params: List[torch.Tensor]) -> Any:
    h1_conv = F.conv2d(input=x, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat


def net_params() -> List[torch.Tensor]:
    scale = 0.01
    w1 = torch.randn(size=(20, 1, 3, 3)) * scale
    b1 = torch.zeros(20)
    w2 = torch.randn(size=(50, 20, 5, 5)) * scale
    b2 = torch.zeros(50)
    w3 = torch.randn(size=(800, 128)) * scale
    b3 = torch.zeros(128)
    w4 = torch.randn(size=(128, 10)) * scale
    b4 = torch.zeros(10)
    params = [w1, b1, w2, b2, w3, b3, w4, b4]
    return params


def get_params(params, device):
    new_params = [param.clone().to(device) for param in params]

    for param in new_params:
        param.attach_grad()

    return new_params


if __name__ == '__main__':
    parameters = net_params()
    loss = nn.CrossEntropyLoss(reduction="none")

    new_params = get_params(parameters, )





