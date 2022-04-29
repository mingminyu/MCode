import random

import matplotlib.pyplot as plt
import torch

__doc__ = "单元线性回归"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b 噪声"""
    X = torch.arange(0, 10, 0.01).reshape(num_examples, -1)
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.5, y.shape)
    return X, y.reshape((-1, 1))


# noinspection DuplicatedCode
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机的，没有特定顺序
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])

        yield features[batch_indices], labels[batch_indices]


def linear_regression(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方误差"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    true_w = torch.tensor([[2.0]])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features: ', features[0], '\nlabel: ', labels[0])

    batch_size = 10
    w = torch.normal(0, 0.01, size=(1, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linear_regression
    loss = squared_loss

    colors = ['r', 'g', 'y']
    plt.scatter(features.numpy(), labels.numpy(), 1)

    for epoch in range(num_epochs):
        # noinspection DuplicatedCode
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

        predictions = torch.matmul(features, w) + b
        plt.plot(features.detach().numpy(), predictions.detach().numpy(), colors[epoch])

    plt.show()
