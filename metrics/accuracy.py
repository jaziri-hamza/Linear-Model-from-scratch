def accuracy(Y, Y_pred):
    
    length = Y_pred.shape[0]
    true_count = 0
    false_count = 0
    
    
    for i in range(Y.shape[0]):
        if( Y[i] == Y_pred[i]):
            true_count +=1
        else:
            false_count +=1
    return true_count / length