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

    epsilon = 10**-10


    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def optimize(self):
        """
            the callable method at each iteration
            to update the new weights
        """
        pass

    def setDerivativeFunction(self, der_function):
        """
            set the derivative function of cost function
            to calcul the derivative of weights
            ...

            Attribut:
            ---------
            der_function: lambda function
        """
        self._der = der_function

    

    