from abc import ABCMeta, abstractmethod
from AbstractClass import BaseAnimal


class Flyable(metaclass=ABCMeta):
    @abstractmethod
    def fly(self):
        pass


class Bird(BaseAnimal, Flyable):
    def eat(self):
        pass

    def fly(self):
        pass

