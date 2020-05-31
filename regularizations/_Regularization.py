import numpy as np

class Regularization():
    
    def __init__(self, cost_func, der_cost_func, regularization_func, der_regularization_func):
        self.setCostFunc( cost_func, regularization_func )
        self.setDerCostFunc( der_cost_func, der_regularization_func )

    def setCostFunc(self, loss_func, regularization_func):
        self._costFunc = lambda W,X,Y: loss_func(W,X,Y) + regularization_func(W)

    def setDerCostFunc(self, der_cost_func, der_regularization_func):
        self._derCostFunc = lambda W,X,Y: der_cost_func(W,X,Y) + der_regularization_func(W)

    def getDerCostFunc(self):
        """
            return a lambda function
        """
        return self._derCostFunc

    def getCostFunc(self):
        """
            return a lambda functio
        """
        return self._costFunc
    