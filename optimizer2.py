import numpy as np 
import datetime
import random

# print("optimizer v2 imported")

class SGD():
    """docstring for SGD"""
    def __init__(self, lr, name='sgd'):
        super(SGD, self).__init__()
        self.lr = lr
        self.name = name

    def initialize(self):
        pass

    def optimize(self,gradients,parameters):
        for i in range(len(parameters)):
            parameters[i] -= lr * gradients[i]
    def __str__(self):
        return self.name + " lr: " + str(self.lr)


class RMSPROP():
    """docstring for RMSPROP"""
    Rs = None
    def __init__(self, lr=0.001, p=0.9, name='rmsprop'):
        super(RMSPROP, self).__init__()
        self.lr = lr
        self.p = p
        self.name = name
        self.delta = 1e-6

    def initialize(self, shapes):
        for shape in shapes:
            assert(type(shape) == tuple)
        self.Rs = []
        for shape in shapes:
            self.Rs.append(np.zeros(shape))


    def optimize(self,gradients,parameters):
        assert(len(gradients) == len(parameters) and len(self.Rs) == len(gradients))
        for i in range(len(self.Rs)):
            self.Rs[i] = self.p * self.Rs[i] + (1 - self.p) * gradients[i] * gradients[i]
            parameters[i] -= (self.lr / np.sqrt(self.delta + self.Rs[i])) * gradients[i]

        
    def __str__(self):
        return self.name + " lr: " + str(self.lr)