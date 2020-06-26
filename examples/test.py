from sys import path
import numpy as np

path.append("../..")

from mylib.metrics import MeanAbsoluteError


# print(MeanAbsoluteError(Y,Y_pred))
from sklearn.datasets import load_breast_cancer, load_wine

data = load_breast_cancer()
X_train = data["data"][0:400, :]
X_test = data["data"][400:-1, :]
Y = np.mat(data["target"]).T
Y_train = Y[0:400, :]
Y_test = Y[400:-1, :]



from mylib.models import NN



m = NN(X_train,Y_train, [3])

m.fit()
