from abc import ABCMeta, abstractmethod


class BaseAnimal(metaclass=ABCMeta):
    color = None

    @abstractmethod
    def eat(self):
        pass


class Tigger(BaseAnimal):
    def eat(self):
        pass


if __name__ == '__main__':
    tigger = Tigger()
