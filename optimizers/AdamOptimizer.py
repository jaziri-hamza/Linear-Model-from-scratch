from mylib.optimizers import Optimizer
import numpy as np

class AdamOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, alpha=0.9, beta=0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        super().__init__()
        
    def optimize_func(self, W, dW, *argv):
        if( not self, hasattr(self, '_ldW')):
            self._ldW1 = self._ldW2 = np.zeros(W.shape)
        iter = int(argv[0])
        print(iter)
        self._ldW1 = self.alpha * self._ldW1 + (1 - self.alpha) * dW
        self._ldW2 = self.beta * self._ldW2 + (1 - self.beta) * np.square(dW)
        curerntldW1 = self._ldW1 / ((1 - self.alpha**iter ) + self.epsilon)
        curerntldW2 = self._ldW2 / ((1 - self.beta**iter) + self.epsilon)
        return W - self.learning_rate * ( curerntldW1 / np.sqrt(curerntldW2 + self.epsilon) )
    