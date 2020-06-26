# shape shoold be  (n,m) with m > 0
def ConfusionMatrix(Y, Y_pred):
    if(Y.shape != Y_pred.shape):
        raise Exception("Y_pred, Y have not the same dimentions")
    result = {
        "True": 0,
        "False": 0,
        "Accuracy": 0
    }
    for i in range(Y_pred.shape[0]):
        if( Y_pred[i,:] == Y[i,:] ):
            result["True"] +=1
        else:
            result["False"] +=1
    result["Accuracy"] = result["True"] / ( result["True"] + result["False"])
    return result