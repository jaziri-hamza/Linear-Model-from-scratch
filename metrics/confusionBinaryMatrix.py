def ConfusionBinaryMatrix(Y, Y_pred):
    result = {
        "TruePositive": 0,
        "TrueNegative": 0,
        "FalsePositive": 0,
        "FalseNegative": 0
    }
    if(Y_pred.shape != Y.shape):
        raise Exception("Y_pred, Y have not the same dimention")
    for i in range(Y.shape[0]):
        if(Y[i,:] == 1):
            if(Y_pred[i,:] == 1):
                result["TruePositive"] +=1
            else:
                result["TrueNegative"] +=1
        else:
            if(Y_pred[i,:]== 0):
                result["FalsePositive"] +=1
            else:
                result["FalseNegative"] +=1
    return result