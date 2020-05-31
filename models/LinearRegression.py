import numpy as np

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
    def __init__(self, X, Y, intercept=False, regularization='', optimizer=''):
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
        

    def fit(self, learning_rate=0.001, iter=100, regularization=None, alpha=0, beta=0, optimizer='gd'):
        """
            it's the method where we will fit the data
            Parameters:
                learning_rate: float
                regularization: string, 'L1'|'L2'|'EL'
                    'L1': Lasso Regression
                    'L2': Ridge Regression
                    'EL': Elastic Net Regression
                alpha, beta: float, the parameters of regularization
                    both arguments used just only when we have a regularization method
                    beta argument just used in case u shoose the 'EL' regularization
                optimizer: string, is the optimizer algorithm
                    'gd': gradient descent
                    'momunto': momunto optimizer
                    'rms': Rootmeansquare optimizer
                    'adam': Adam optimizer
        """
        if regularization:
            self._alpha = alpha
            self._beta = beta
        loss = self.__lossFunction(regularization=regularization)
        lossDerivative = self.__lossFunctionDerivative(regularization=regularization)
        # batch gradient
        for iter in range(iter):
            dW = lossDerivative(self._W, self._X, self._Y)
            self._W = self.AdamOptimizer(dW, 0.8, 0.8, iter, learning_rate)
            self._costs.append(loss(self._W, self._X, self._Y))

    def predict(self, X_test):
        """
            it's the method where we get the prediction of inputs with actual weights
            Parameters:
                X_test: the input test
        """
        if(self._intercept):
            X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
        return np.dot(X_test, self._W)

   

    def __lossFunction(self, regularization=None):
        """
            return function to calcul the loss function,
            for every regularization parameter type
            we use the <MeanSquareError> function
        """
        error = lambda W, X, Y: ( (np.sum( np.square( np.dot(X,W) - Y ) )) ) / 2 * X.shape[0]
        if regularization == 'L1' :
            return lambda W, X, Y: error(W,X,Y) + + (self._alpha * np.sum( np.abs(X) ))
        elif regularization == 'L2':
            return lambda W, X, Y: error(W,X,Y) + (self._alpha * np.sum( np.square(X) ))
        elif regularization == 'EL':
            return lambda W, X, Y: error(W,X,Y) + self._alpha * (  (1-self._beta / 2) * np.sum( np.square(W) ) + (self._beta * np.sum( np.abs(W) )) )
        else:
            return error

    def __lossFunctionDerivative(self, regularization=None):
        """
            return the derivative of loss function <MeanSquareError> ,
            for every regularzation parameter type
        """
        derivativeError = lambda W, X, Y: np.dot( X.T, ( np.dot(X,W) - Y ) )
        if regularization == 'L1':
            return lambda W,X,Y: derivativeError(W,X,Y) + ( self._alpha * np.sign(W) )
        elif regularization == 'L2':
            return lambda W,X,Y: derivativeError(W,X,Y) + (2 * self._alpha * W)
        elif regularization == 'EL':
            return lambda W, X, Y: derivativeError(W,X,Y) + self._alpha * ( (self._beta * np.sign(W)) + (2 * self._beta * W) )
        else:
            return derivativeError

    def gradientDescentOptimizer(self, dW, learning_rate):
        return self._W - learning_rate * dW

    def momuntoOptimizer(self, dW, beta, learning_rate):
        if(not hasattr(self, '_vdW')):
            self._vdW = np.zeros(dW.shape)
        self._vdW =  beta * self._vdW + (1 - beta) * dW
        return self._W - learning_rate * self._vdW

    def RMSPropOptimizer(self, dW, beta, learning_rate):
        """
            RootMeanSquareProp optimizer
        """
        epsilon = 0.00000001
        if( not hasattr(self, '_vdw')):
            self._vdW = np.zeros(dW.shape)
        self._vdW = beta * self._vdW + ((1- beta) * np.square(dW))
        return self._W - learning_rate * (dW/np.sqrt(self._vdW + epsilon ))

    
    def AdamOptimizer(self, dW, beta1, beta2, iter, learning_rate):
        """
            
        """
        epsilon = 0.00000001
        if(not hasattr(self, '_mvdw')):
            self._mvdw = np.zeros(self._W.shape)
            self._svdw = np.zeros(self._W.shape)
        self._mvdw = beta1 * self._mvdw + (1 - beta1) * dW
        self._svdw = beta2 * self._svdw + (1 - beta2) * np.square(dW)
        mvdwC = self._mvdw / ((1 - beta1**iter ) + epsilon)
        svdwC = self._svdw / ((1 - beta2**iter) + epsilon)
        return self._W - learning_rate * ( mvdwC / np.sqrt(svdwC + epsilon) )