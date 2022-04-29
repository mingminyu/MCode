import random

import matplotlib.pyplot as plt
import tensorflow as tf

__doc__ = "多元线性回归"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b 噪声"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))

    return X, y


# noinspection DuplicatedCode
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机的，没有特定顺序
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])

        yield tf.gather(features, j), tf.gather(labels, j)


# noinspection DuplicatedCode
def linear_regression(X, w, b):
    """线性回归模型"""
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方误差"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, grads, lr, batch_size):
    """小批量随机梯度下降"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


if __name__ == '__main__':
    true_w = tf.constant([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features: ', features[0], '\nlabel: ', labels[0])

    batch_size = 10
    w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(1), trainable=True)

    lr = 0.03
    num_epochs = 3
    net = linear_regression
    loss = squared_loss

    for epoch in range(num_epochs):
        # noinspection DuplicatedCode
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = loss(net(X, w, b), y)

            dw, db = g.gradient(l, [w, b])
            sgd([w, b], [dw, db], lr, batch_size)

        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

    plt.scatter(features[:, 1], labels, 1)
    plt.show()
