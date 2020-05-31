import numpy as np
from mylib.regularizations import Regularization

class ElasticNetRegularization(Regularization):

    def __init__(self,  alpha=0.8, beta=0.8):
        self._alpha = alpha
        self._beta = beta

    def _init_reg(self, cost_func, der_cost_func):
        super().__init__(
            cost_func, der_cost_func,
            lambda W :  self._alpha * (  (1-self._beta / 2) * np.sum( np.square(W) ) + (self._beta * np.sum( np.abs(W) )) ), # regularization func
            lambda W:   self._alpha * ( (self._beta * np.sign(W)) + (2 * self._beta * W) ) # derivate of regularization func
            )