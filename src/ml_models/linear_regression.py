import numpy as np
from sklearn.metrics import f1_score, accuracy_score

class LinearRegression:
    def fit(self, _X, _y):
        X = np.array(list(map(lambda x: [1, x[0], x[1]], _X)))


        X =  np.array(X)
        y =  np.array(_y)
        xTx = np.dot(X.transpose(), X)
        inverse = np.linalg.inv(xTx)
        self.w = np.dot(np.dot( inverse, X.transpose()), y)
    
    def predict(self, _x):

        X = np.array(list(map(lambda x: [1, x[0], x[1]], _x)))

        return [np.sign(np.dot(self.w, xn)) for xn in X]
    
    def getOriginalY(self, originalX):
        return (-self.w[0] - self.w[1]*originalX) / self.w[2]

    def get_w(self):
        return self.w

    def score(self, X, Y):
        return accuracy_score(Y, self.predict(X))