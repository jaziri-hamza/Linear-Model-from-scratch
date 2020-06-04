from mylib.optimizers import Optimizer
import numpy as np

class MomuntoOptimizer(Optimizer):

    def __init__(self, learning_rate=0.001, alpha=0.9):
        self.learning_rate = learning_rate
        self.alpha = alpha
        super().__init__()
        
    def optimize_func(self, W, dW, *argv):
        if( not self, hasattr(self, '_ldW')):
            self._ldW = np.zeros(W.shape)
        self._ldW = self.alpha * self._ldW + (1 - self.alpha) * dW
        return  W - self.learning_rate * self._ldW
    