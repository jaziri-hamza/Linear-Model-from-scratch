import numpy as np

"""
    is a class used to normalize the range of independent variables or features of data
    ...
    Attributs:
        X : numpy.ndarray
            the data went to normalize
"""
class MeanNormalization:
    def __init__(self, X):
        if( not isinstance(X, np.ndarray)):
            raise Exception( str( type(X)) + " is not instance from numpy.ndarray")
        self._min = np.amin(X)
        self._max = np.amax(X)
        self._mean = np.mean(X) # average
        self._X = X

    def scale(self):
        return (self._X - self._mean ) / (self._max - self._min)

