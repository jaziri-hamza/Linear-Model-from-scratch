import numpy as np

class Regularization():
    
    def __init__(self, cost_func, der_cost_func, regularization_func, der_regularization_func):
        self.setCostFunc( cost_func, regularization_func )
        self.setDerCostFunc( der_cost_func, der_regularization_func )

    def setCostFunc(self, loss_func, regularization_func):
        self._costFunc = loss_func + regularization_func

    def setDerCostFunc(self, der_cost_func, der_regularization_func):
        self._derCostFunc = der_cost_func + der_regularization_func

    def calcDerCostFunc(self, W, X, Y):
        return self._derCostFunc(W,X,Y)

    def calcCostFunc(self, W, X, Y):
        """
        """
        return self._costFunc(W, X, Y)
    