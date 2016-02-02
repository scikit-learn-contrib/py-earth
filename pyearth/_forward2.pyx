
import numpy as np


cdef class ForwardPasser:
    
    def  __init__(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X,
                 cnp.ndarray[BOOL_t, ndim=2] missing,
                 cnp.ndarray[FLOAT_t, ndim=2] y,
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight,
                 **kwargs):
        
        self.sample_weight = np.sqrt(sample_weight)
        self.y = y
        self.X = X
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.n_outcomes = self.y.shape[1]
        self.n_weights = self.sample_weight.shape[1]
        self.missing = missing
        self.process_configuration_args(kwargs)
        
        if self.allow_missing:
            self.has_missing = np.any(self.missing, axis=0).astype(BOOL)
        
        
    def process_configuration_args(self, kwargs):
        self.endspan       = kwargs.get('endspan', -1)
        self.minspan       = kwargs.get('minspan', -1)
        self.endspan_alpha = kwargs.get('endspan_alpha', .05)
        self.minspan_alpha = kwargs.get('minspan_alpha', .05)
        self.max_terms     = kwargs.get('max_terms', 2 * self.n + 10)
        self.allow_linear  = kwargs.get('allow_linear', True)
        self.max_degree    = kwargs.get('max_degree', 1)
        self.thresh        = kwargs.get('thresh', 0.001)
        self.penalty       = kwargs.get('penalty', 3.0)
#         self.check_every   = kwargs.get('check_every', -1)
        self.min_search_points = kwargs.get('min_search_points', 100)
        self.xlabels       = kwargs.get('xlabels')
        self.use_fast = kwargs.get('use_fast', False)
        self.fast_K = kwargs.get("fast_K", 5)
        self.fast_h = kwargs.get("fast_h", 1)
        self.zero_tol = kwargs.get('zero_tol', 1e-12)
        self.allow_missing = kwargs.get("allow_missing", False)
        if self.endspan < 0:
            self.endspan = round(3 - log2(self.endspan_alpha / self.n))
        
    def init_linear_variables(self):
        root_basis_function = self.basis[0]
        for variable in range(self.n):
            
        
        
        