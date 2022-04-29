import torchvision
from torch.utils import data
from torchvision import transforms


def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
    return [text_labels[int(i)] for i in labels]


def fashion_mnist_dataset():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=trans)

    print("Train: ", len(mnist_train), "\nTest: ", len(mnist_test))


def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, download=True, transform=trans)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, download=True, transform=trans)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4))
