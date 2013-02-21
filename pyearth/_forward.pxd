
cimport numpy as cnp
import numpy as np
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t
from _basis cimport Basis

ctypedef enum StoppingCondition:
    MAXTERMS=0,
    MAXRSQ=1,
    NOIMPRV=2,
    LOWGRSQ=3,
    NUMDIFF=4,
    NUMERR=5
    

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
    cdef ForwardPassRecord record
    cdef cnp.ndarray X
    cdef cnp.ndarray y
    cdef unsigned int m
    cdef unsigned int n
    cdef Basis basis
    cdef cnp.ndarray B
    cdef list xlabels
    cdef cnp.ndarray sort_tracker
    cdef cnp.ndarray sorting
    cdef cnp.ndarray mwork
    cdef cnp.ndarray delta
    cdef cnp.ndarray R
    cdef cnp.ndarray u
    cdef cnp.ndarray v
    
    cpdef Basis get_basis(ForwardPasser self)
    
    cpdef run(ForwardPasser self)
    
    cdef stop_check(ForwardPasser self)
    
    cdef next_pair(ForwardPasser self)
    
    cdef best_knot(ForwardPasser self, unsigned int parent, unsigned int variable, cnp.ndarray[INT_t,ndim=1] candidates, FLOAT_t * mse, FLOAT_t * knot, unsigned int * knot_idx)
        
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
    
    cpdef FLOAT_t get_mse(ForwardPassIteration self)
    
    cpdef unsigned int get_size(ForwardPassIteration self)
    
cdef class FirstForwardPassIteration(ForwardPassIteration):
    cpdef unsigned int get_size(FirstForwardPassIteration self)

    
    
    
    
    