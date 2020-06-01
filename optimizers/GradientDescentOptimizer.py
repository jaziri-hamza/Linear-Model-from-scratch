from mylib.optimizers import Optimizer


class GradientDescentOptimizer(Optimizer):
    
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        super().__init__()
        
    def optimize_func(self, W, dW, *argv):
        return  W - ( self.learning_rate * dW )