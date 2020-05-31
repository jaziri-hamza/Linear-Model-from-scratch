import numpy as np
from mylib.regularizations import Regularization

class LassoRegularization(Regularization):

    def __init__(self, alpha=0.8):
        self._alpha = alpha

    def _init_reg(self, cost_func, der_cost_func):
        super().__init__(
            cost_func, der_cost_func,
            lambda W : self._alpha * np.sum( np.abs(W) ), # regularization func
            lambda W:  self._alpha * np.sign(W) # derivate of regularization func
            )