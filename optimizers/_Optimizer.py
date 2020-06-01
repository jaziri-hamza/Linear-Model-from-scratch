import numpy as np

class Optimizer:
    """
        Optimizer is the base class of any optimizer algorithm
        where we will tell the model how to update the weights
        ...

        Attributs:
        learning_rate: float, Default 0.001
            is the step size while moving toward a minimun
            of cost function
        
    """

    # small number we need it for some operation mathematique
    epsilon = 10**-10


    def __init__(self):
        pass

    def optimize(self, W, dW, *argv):
        """
            the callable method at each iteration
            to update the new weights
            ...

            Attributs
            ---------
            dW: np.darray
                is a matrix contains the derivative of weights
        """
        # call the function where tell us how to update the weight
        return self.optimize_func(W, dW, *argv)