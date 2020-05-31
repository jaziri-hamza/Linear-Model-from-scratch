import random

def split_into_batchs(batch_size, X , Y):
        """
            split the inputs and targets into mini batchs
            we will use a random function for get a mix data
            return:
                Array of Array<ofInputs>
                Array of Array<ofTarget>
        """
        items_number_of_each_batch = int(X.shape[1] / batch_size)
        if(items_number_of_each_batch == 0):
            items_number_of_each_batch = 1
        indices = []
        for i in range(X.shape[0]):
            indices.append(i)

        X_batchs = []
        Y_batchs = []
        for i in range(batch_size):
            X_batchs.append([])
            Y_batchs.append([])
            for j in range(items_number_of_each_batch):
                randIndice = random.randint(0, len(indices)-1)
                X_batchs[i].append(X[ indices[randIndice], : ])
                Y_batchs[i].append(Y[ indices[randIndice], : ])
                indices.__delitem__(randIndice)
        
        # add the rest of inputs in new batch
        if(len(indices)>0):
            X_batchs.append([])
            Y_batchs.append([])
            for i in range(len(indices)):
                X_batchs[len(X_batchs)-1].append(X[indices[i], :])
                Y_batchs[len(X_batchs)-1].append(Y[indices[i], :])
        return X_batchs, Y_batchs