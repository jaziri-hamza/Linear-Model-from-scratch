import numpy as np

def MeanAbsoluteError(Y_pred, Y):
    if(Y_pred.shape != Y.shape):
        raise Exception("Y_pred, Y have not the same dimension")
    if(Y_pred.shape[0] == 1):
        length = Y_pred.shape[1]
    else:
        length = Y_pred.shape[0]
    return 1/length * np.sum( np.abs(Y_pred - Y) )