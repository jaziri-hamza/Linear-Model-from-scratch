import numpy as np
from mylib.optimizers import GradientDescentOptimizer

class LinearRegression():
    """
        LinearRegression is useful for finding the relationship
        between inputs and target
        where we can find the best line to fit the inputs data
        ...

        Attributes
        ----------
        X : numpy.darray
            the inputs data, with shapes (n_simples, n_features)
        Y : numpy.darray
            the targets label, with shapes (n_simples,1)
        intercept: boolean, optional, default False
            the intercept of the model, 
            if set to false, no intercept will be calculated
            (example): eq = ax + b : the variable b is the intercept
        ...

        Example
        -------
        >>> import numpy as np
        >>> from mylib.models import LinearRegression
        >>> X = np.linspace(10, 100, 100).reshape(-1, 1)
        >>> Y = np.linspace(10, 100, 100).reshape(-1, 1)
        >>> lin_reg = LinearRegression(X, Y, intercept=True)
        >>> lin_reg.fit()
    """
    def __init__(self, X, Y, intercept=False, regularization=None, optimizer=None):
        self._intercept = intercept
        if(intercept):
            self._X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        else:
            self._X = X
        self._Y = Y
        self.n_simples = self._X.shape[0]
        self.n_features = self._X.shape[1]
        self._W = np.zeros((self.n_features, 1))
        self._costs = []
        self._init_func(regularization)
        if(optimizer):
            self._optimizer = optimizer
        else:
            # default optimizer
            self._optimizer = GradientDescentOptimizer(0.01)
        

    def fit(self, learning_rate=0.001, iter=100):
        """
            it's the method where we will fit the data
            Parameters:
                learning_rate: float
                optimizer: string, is the optimizer algorithm
                    'gd': gradient descent
                    'momunto': momunto optimizer
                    'rms': Rootmeansquare optimizer
                    'adam': Adam optimizer
        """
        # batch gradient
        for iter in range(iter):
            dW = self._der_cost_func(self._W, self._X, self._Y)
            self._W = self._optimizer.optimize(self._W, dW, iter)
            # self._W = self.AdamOptimizer(dW, 0.8, 0.8, iter, learning_rate)
            self._costs.append(self._cost_func(self._W, self._X, self._Y))

    def predict(self, X_test):
        """
            it's the method where we get the prediction of inputs with actual weights
            Parameters:
                X_test: the input test
        """
        if(self._intercept):
            X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
        return np.dot(X_test, self._W)

    def _init_func(self, regularization):
        self._cost_func = lambda W, X, Y: ( (np.sum( np.square( np.dot(X,W) - Y ) )) ) / 2 * X.shape[0]
        self._der_cost_func = lambda W, X, Y: np.dot( X.T, ( np.dot(X,W) - Y ) )
        if(regularization):
            regularization._init_reg(
                self._cost_func,
                self._der_cost_func
            )
            self._cost_func = regularization.getCostFunc()
            self._der_cost_func = regularization.getDerCostFunc()