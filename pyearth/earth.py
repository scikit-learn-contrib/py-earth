'''
Created on Feb 16, 2013

@author: jasonrudy
'''
from _forward import ForwardPasser



class Earth(object):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.forward_pass(X, y)
        self.prune(X, y)
        self.linear_fit()
    
    def forward_pass(self, X, y):
        forward_passer = ForwardPasser(X, y)
        forward_passer.run()
        self.basis = forward_passer.get_basis()
    
    def prune(self, X, y):
        pass
    
    def ols(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def transform(self, X):
        pass
    
class Pruner(object):
    def __init__(self, basis, X, y):
        pass
    
    def run(self):
        pass
    


        
    
    
    
    

