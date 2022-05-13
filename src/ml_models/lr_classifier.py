from linear_regression import LinearRegression
from pocketpla import PocketPLA

class LRClassifier():
    def execute(self, _X, _y):
        lr = LinearRegression()
        lr.execute(_X, _y)
        self.w = lr.get_w()
        
        #pla = PLA()
        #pla.set_w(self.w)
        #pla.execute(_X, _y)
        #self.w = pla.get_w()
        
        
    def predict(self, x):
        return np.sign(np.dot(self.w, x))
     
    def getRegressionY(self, regressionX, shift=0):
        return (-self.w[0]+shift - self.w[1]*regressionX) / self.w[2]