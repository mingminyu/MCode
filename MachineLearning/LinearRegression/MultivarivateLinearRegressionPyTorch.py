import torch
from torch import nn
from torch.utils import data


__doc__ = "多元线性回归简洁版"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b 噪声"""
    X = torch.normal(0, 0.1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 10
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()

        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
