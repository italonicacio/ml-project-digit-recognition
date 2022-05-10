import numpy as np
from sklearn.metrics import f1_score, accuracy_score


class PocketPLA:

    def __init__(self) -> None:
        pass

    def fit(self, x_train, y_train, max_iter = 1500, epsilon = 1e-6):

        x = np.array(list(map(lambda x: [1, x[0], x[1]], x_train)))
        y = y_train
        w = np.zeros(len(x[0]))


        best_ein = len(y)
        best_w = w

        it = 0
        e = 0.0
        while it < max_iter:
            
            for x_i, y_i in zip(x, y):

                if np.sign(np.dot(w, x_i)) != y_i:
                    w = w + y_i*x_i
                    e_in = self.Ein(w,x,y)
                

                    if best_ein > e_in:
                    

                        e = np.abs(best_ein - e_in)
                        best_ein = e_in
                        best_w = w
            
            it += 1



        self.w = best_w

    def Ein(self, w, x, y):
        e_in = 0
        for x_i, y_i in zip(x, y):
            h = np.sign(np.dot(w, x_i))
            if h != y_i:
                e_in += 1
            
        return e_in


    def predict(self, x_test):
        
        x = list(map(lambda x: np.array([1, x[0], x[1]]), x_test))
        
        predict_y = list(map(lambda i: np.sign(np.dot(self.w, i)), x))

        return predict_y

    def get_w(self):
        return self.w

    def set_w(self, new_w):
        self.w = new_w

    def score(self, X, Y):

        return accuracy_score(Y, self.predict(X))
