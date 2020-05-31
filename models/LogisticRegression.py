import numpy as np

from mylib.utils import split_into_batchs
from mylib.utils import sigmoid

class LogisticRegression:
    def __init__(self, X, Y, intercept=False):
        """
            it's the constructor method where we will initialize all variables we need
            Parameters:
                X: np.darray, inputs, the shape shoold be (training_examples, features)
                Y: np.darray, targets
                intercept: boolean, the bias parameter
        """
        self._intercept = intercept
        if(intercept):
            self._X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        else:
            self._X = X
        self._Y = Y
        self._simples_number = self._X.shape[0]
        self._features_number = self._X.shape[1]
        self._W = np.zeros((self._features_number, 1))
        self._costs = []

    def fit(self, learning_rate=0.001, batch_size=0, epoch=100, regularization=None, alpha=0, beta=0):
        """
            it's the method where we will fit the data
            Parameters:
                learning_rate: float
                batch_size: number, shoold be less than the dataset size
                epoch: number, the number of iteration for full dataset
                regularization: string, 'L1'|'L2'|'EL'
                    'L1': Lasso Regression
                    'L2': Ridge Regression
                    'EL': Elastic Net Regression
                alpha, beta: float, the parameters of regularization
                    both arguments used just only when we have a regularization method
                    beta argument just used in case u shoose the 'EL' regularization
        """
        if regularization:
            self._alpha = alpha
            self._beta = beta
        loss = self.__lossFunction(regularization=regularization)
        gd = self.__gradientDescent(regularization=regularization)
        if(batch_size > 0):
            # mini batch gradient
            X_batch, Y_batch = self.split_into_batchs(batch_size)
            for iter in range(epoch):
                for i in range(len(X_batch)):
                    self._W -= learning_rate * gd(self._W, np.array(X_batch[i]), np.array(Y_batch[i]))
                    self._costs.append(loss(self._W,np.array(X_batch[i]),np.array(Y_batch[i])))
        else:
            # batch gradient
            for iter in range(epoch):
                self._W -= learning_rate * gd(self._W, self._X, self._Y)
                self._costs.append(loss(self._W, self._X, self._Y))


    def predict(self):
        pass

    def __decision(self):
        pass

    def __lossFunction(self, regularization=None):
        """
            return function to calcul the loss function,
            for every regularization parameter type
            we use the <MeanSquareError> function
        """
        error = lambda W, X, Y: np.sum( -Y * np.log( sigmoid( np.dot(X,W) ) ) - (1- Y) * np.log( 1 - sigmoid( np.dot(X,W) ) ) )
        if regularization == 'L1' :
            return lambda W, X, Y: error(W,X,Y) + + (self._alpha * np.sum( np.abs(X) ))
        elif regularization == 'L2':
            return lambda W, X, Y: error(W,X,Y) + (self._alpha * np.sum( np.square(X) ))
        elif regularization == 'EL':
            return lambda W, X, Y: error(W,X,Y) + self._alpha * (  (1-self._beta / 2) * np.sum( np.square(W) ) + (self._beta * np.sum( np.abs(W) )) )
        else:
            return error

    def __gradientDescent(self, regularization=None):
        """
            return the derivative of loss function <MeanSquareError> ,
            for every regularzation parameter type
        """
        derivativeError = lambda W, X, Y: np.dot( X.T, ( np.dot(X,W) - Y ) ) / X.shape[0]
        if regularization == 'L1':
            return lambda W,X,Y: derivativeError(W,X,Y) + ( self._alpha * np.sign(W) )
        elif regularization == 'L2':
            return lambda W,X,Y: derivativeError(W,X,Y) + (2 * self._alpha * W)
        elif regularization == 'EL':
            return lambda W, X, Y: derivativeError(W,X,Y) + self._alpha * ( (self._beta * np.sign(W)) + (2 * self._beta * W) )
        else:
            return derivativeError