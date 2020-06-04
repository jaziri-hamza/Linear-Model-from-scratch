import numpy as np

from mylib.optimizers import GradientDescentOptimizer
from mylib.utils import sigmoid

class LogisticRegression:
   
    def __init__(self, X, Y, intercept=False):
        self._intercept = intercept
        if(intercept):
            self._X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        else:
            self._X = X
        self._Y, self._target_count = self._transform_target(Y)
        self.n_simples = self._X.shape[0]
        self.n_features = self._X.shape[1]
        self._W = np.zeros((self.n_features, self._target_count))

    # def _transform_back_target(self, Y_transormed):


    def _transform_target(self, Y):
        """
            transform target labels for multiclassification
            if it's a binary classification just return the same Y
            
            Example
            -------
            given Y = [0,1,2,3,1,2]
            return [
                    [0, 0, 0, 0], # 0
                    [0, 1, 0, 0], # 1
                    [0, 0, 0, 1], # 3
                    [0, 1, 0, 0], # 1
                    [0, 0, 1, 0], # 2
            ], 4 # unique_target_count
        """
        unique_target = np.unique(Y)
        if(len(unique_target) <= 2):
            return Y, 1
        else:
            Y_updated = np.zeros( (Y.shape[0], len( unique_target ) ) )
            for i in range((Y.shape[0])):
                Y_updated[i, :] = self._row_encoder(len(np.unique(Y)),  np.squeeze(Y[i]) )
        return Y_updated, len(unique_target)

    def _transform_back_target(self, Y):
        """
            transform predected labels structure like the first one
            simply is the reverse of transform_target
        """
        if( self._target_count < 2 ):
            Y_new = Y > .5
            return Y_new.astype(int)
        else:
            Y_new = np.zeros((Y.shape[0], 1))
            for i in range(Y.shape[0]):
                Y_new[i, :] = np.where( Y[i,:] == np.amax(Y[i,:]) )
            return Y_new

    def _row_encoder(self, length, nbr):
        """
            the encoder foreach row in target label
            example
            -------
            max Number in target label is 7 as 'length', 
            give a 2 as 'nbr' => return [ 0 0 1 0 0 0 0 ]
        """
        A = np.zeros((1, length))
        for i in range(length):
            if(i == nbr):
                A[:,i] = 1
            else:
                A[:,i] = 0
        return A
    

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
        # print(self._W.shape)
        return self._transform_back_target( sigmoid(np.dot(X_test, self._W)) )
    
    def _init_func(self, regularization):
        """
            here we will initialize 
            the cost function and
            the derivative of the cost function
            ...
        """
        # default cost function
        self._cost_func = lambda W,X,Y : np.sum( np.square( (Y * np.log( sigmoid( np.dot(X,W) ) ) ) + ( (1-Y) * np.log( 1 - sigmoid( np.dot(X,W) )) ) ) )
        self._der_cost_func = lambda W, X, Y: np.dot( X.T, ( sigmoid(np.dot(X,W)) - Y ) )
        if(regularization):
            regularization._init_reg(
                self._cost_func,
                self._der_cost_func
            )
            self._cost_func = regularization.getCostFunc()
            self._der_cost_func = regularization.getDerCostFunc()


    def boundary_descion(self, Y):
        return Y