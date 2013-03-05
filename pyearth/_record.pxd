cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.int_t INT_t
ctypedef cnp.uint8_t BOOL_t
from _basis cimport Basis

cdef class Record:
    cdef list iterations
    cdef int num_samples
    cdef int num_variables
    cdef FLOAT_t penalty
    cdef FLOAT_t sst #Sum of squares total
    
    cpdef append(Record self, Iteration iteration)
    
    cpdef FLOAT_t mse(Record self, unsigned int iteration)
    
    cpdef FLOAT_t rsq(Record self, unsigned int iteration)
    
    cpdef FLOAT_t gcv(Record self, unsigned int iteration)
    
    cpdef FLOAT_t grsq(Record self, unsigned int iteration)

cdef class PruningPassRecord(Record):
    cdef unsigned int selected

    cpdef set_selected(PruningPassRecord self, unsigned int selected)

    cpdef unsigned int get_selected(PruningPassRecord self)

    cpdef roll_back(PruningPassRecord self, Basis basis)
	
cdef class ForwardPassRecord(Record):
    cdef int stopping_condition
    
    cpdef set_stopping_condition(ForwardPassRecord self, int stopping_condition)
    
cdef class Iteration:
    cdef FLOAT_t mse
    cdef unsigned int size
    
    cpdef FLOAT_t get_mse(Iteration self)
    
    cpdef unsigned int get_size(Iteration self)
    
cdef class PruningPassIteration(Iteration):
    cdef unsigned int pruned
    
    cpdef unsigned int get_pruned(PruningPassIteration self)
    
cdef class FirstPruningPassIteration(PruningPassIteration):
    pass
    
cdef class ForwardPassIteration(Iteration):
    cdef unsigned int parent
    cdef unsigned int variable
    cdef FLOAT_t knot
    cdef int code
    
cdef class FirstForwardPassIteration(ForwardPassIteration):
    cpdef unsigned int get_size(FirstForwardPassIteration self)