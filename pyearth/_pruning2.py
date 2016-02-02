'''
Created on Jan 20, 2016

@author: jason
'''



class PruningPasser(object):
    def __init__(self, basis, X, missing, y,
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight,
                 cnp.ndarray[FLOAT_t, ndim=1] output_weight,
                 **kwargs):):