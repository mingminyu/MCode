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


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 TensorFlow 构造器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)

    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    true_w = tf.constant([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    print('features: ', features[0], '\nlabel: ', labels[0])

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01))
    b = tf.Variable(tf.zeros(1), trainable=True)

    lr = 0.03
    num_epochs = 3

    initializer = tf.initializers.RandomNormal(stddev=0.01)
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

    loss = tf.keras.losses.MeanSquaredError()
    trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

    for epoch in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as tape:
                l = loss(net(X, training=True), y)

            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))

        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net.get_weights()[0]
    print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
    b = net.get_weights()[1]
    print('b的估计误差：', true_b - b)
