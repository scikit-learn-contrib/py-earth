
cimport numpy as np
ctypedef np.float64_t FLOAT_t
from _basis cimport Basis

cdef class ForwardPasser:
    cdef int endspan
    cdef int minspan
    cdef FLOAT_t endspan_alpha
    cdef FLOAT_t minspan_alpha
    cdef int max_terms
    cdef int max_degree
    cdef FLOAT_t thresh
    cdef FLOAT_t penalty
    cdef int check_every
    cdef int min_search_points
    cdef list record
    cdef np.ndarray X
    cdef np.ndarray y
    cdef Basis basis
    
    cpdef run(ForwardPasser self)
    
    cdef next_pair(ForwardPasser self)
    
    cdef best_knot(ForwardPasser self)
    
cdef class ForwardPassRecord:
    cdef int stopping_condition
    cdef list iterations
    cdef int num_samples
    cdef int num_variables
    cdef FLOAT_t penalty
    cdef FLOAT_t sst #Sum of squares total
    
    cpdef append(ForwardPassRecord self, ForwardPassIteration iteration)
    
    cpdef set_stopping_condition(ForwardPassRecord self, int stopping_condition)
    
    cpdef FLOAT_t mse(ForwardPassRecord self, unsigned int iteration)
    
    cpdef FLOAT_t rsq(ForwardPassRecord self, unsigned int iteration)
    
    cpdef FLOAT_t gcv(ForwardPassRecord self, unsigned int iteration)
    
    cpdef FLOAT_t grsq(ForwardPassRecord self, unsigned int iteration)
    
    
cdef class ForwardPassIteration:
    cdef unsigned int parent
    cdef unsigned int variable
    cdef FLOAT_t knot
    cdef FLOAT_t mse
    cdef unsigned int size
    cdef int code
    
    
    
    
    