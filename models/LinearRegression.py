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
        >>> X = np.linspace(10, 100, 10).reshape(-1,1)
        >>> Y = np.linspace(10, 100, 10).reshape(-1,1)
        >>> lin_reg = LinearRegression(X, Y)
        >>> lin_reg.fit()
        >>> # draw a simple figure with matplotlib
        >>> import matplotlib.pyplot as plt
        >>> subplot = plt.subplot()
        >>> subplot.scatter(X_,Y_, c="red", marker=".")
        >>> subplot.plot(X_, lr.predict(X_))
        >>> plt.show()
        ---
            # see the "jyputer-examples" foder
            # if you want to see more examples
            # with custom regularization and optimization
    """
    def __init__(self, X, Y, intercept=False):
        self._intercept = intercept
        if(intercept):
            self._X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        else:
            self._X = X
        self._Y = Y
        self.n_simples = self._X.shape[0]
        self.n_features = self._X.shape[1]
        self._W = np.zeros((self.n_features, 1))

    def fit(self, learning_rate=0.00001, iter=100, optimizer=None, regularization=None, history=False):
        """
            simply this method used for training the data
            and update the weights on each iteration
            ...

            Attribut
            --------
                learning_rate: float, Default 0.00001
                    the step size at each iteration for
                    get the minimun cost
                optimizer: Instance of Optimizer CLass, Default GradientDescentOptimizer
                    it's the algorthim to optimize your learning
                    check Optimizer Class Documentation for know all 
                    about the algorithme exist
                regularization: Instance of Regularization CLass, Default None
                    it's the additionnal operation of cost function
                    generally used for solve the overfitting/underfitting problem
                history: boolean, Default False
                    if set to True, that's mean we will save the Error Value
                    of each iteration in array
                    * to get the values of the Errors 
                        >>>> instance.history : return Array<float>
        """
        if(history):
            self.history = []
        
        # init the regularization function
        self._init_func(regularization)

        # init the optimizer function
        if(optimizer):
            self._optimizer = optimizer
        else:
            # default optimizer
            self._optimizer = GradientDescentOptimizer(learning_rate)
        

        for iter in range(iter):
            dW = self._der_cost_func(self._W, self._X, self._Y)
            self._W = self._optimizer.optimize(self._W, dW, iter)
            if(history):
                self.history.append(self._cost_func(self._W, self._X, self._Y))

    def predict(self, X_test):
        """
            predict the outptut of your
            test dataset with current Weights
            return the predict labels

            *   this method used after fitting the data
                and get the optimum weights
        """
        if(self._intercept):
            X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
        return np.dot(X_test, self._W)

    def _init_func(self, regularization):
        """
            here we will initialize 
            the cost function and
            the derivative of the cost function
            ...
        """
        # default cost function
        self._cost_func = lambda W, X, Y: ( (np.sum( np.square( np.dot(X,W) - Y ) )) ) / 2 * X.shape[0]
        self._der_cost_func = lambda W, X, Y: np.dot( X.T, ( np.dot(X,W) - Y ) )
        if(regularization):
            regularization._init_reg(
                self._cost_func,
                self._der_cost_func
            )
            self._cost_func = regularization.getCostFunc()
            self._der_cost_func = regularization.getDerCostFunc()