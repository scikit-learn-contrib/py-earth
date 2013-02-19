# distutils: language = c++

cimport numpy as np
ctypedef np.float64_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef np.uint8_t BOOL_t

cdef class BasisFunction:
    '''Abstract.  Subclasses must implement the apply, translate, scale, and __init__ methods.'''
    
    cdef BasisFunction parent
    cdef dict child_map
    cdef list children
    cdef bint pruned
    cdef bint prunable
    
    cpdef bint has_knot(BasisFunction self)
    
    cpdef bint is_prunable(BasisFunction self)
    
    cpdef bint is_pruned(BasisFunction self)
    
    cdef list get_children(BasisFunction self)
    
    cpdef _set_parent(self,BasisFunction parent)
    
    cpdef _add_child(self,BasisFunction child)
        
    cpdef prune(self)
        
    cpdef unprune(self)
        
    cpdef knots(BasisFunction self, unsigned int variable)
    
    cpdef unsigned int degree(BasisFunction self)
    
    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = ?)
        
    cpdef np.ndarray[INT_t, ndim=1] valid_knots(BasisFunction self, np.ndarray[FLOAT_t,ndim=1] values, np.ndarray[FLOAT_t,ndim=1] variable, int variable_idx, unsigned int check_every, int endspan, int minspan, FLOAT_t minspan_alpha, unsigned int n, np.ndarray[INT_t,ndim=1] workspace)
    

cdef class ConstantBasisFunction(BasisFunction):
    
    cpdef unsigned int degree(ConstantBasisFunction self)
    
    cpdef translate(ConstantBasisFunctionself, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse)
    
    cpdef FLOAT_t scale(ConstantBasisFunctionself, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts)
    
    cpdef _set_parent(self,BasisFunction parent)

    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = ?)
            
    
cdef class HingeBasisFunction(BasisFunction):
    cdef FLOAT_t knot
    cdef unsigned int knot_idx
    cdef unsigned int variable
    cdef bint reverse
    cdef str label
    
    cpdef bint has_knot(HingeBasisFunction self)
    
    cpdef translate(HingeBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse)
            
    cpdef FLOAT_t scale(HingeBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts)
    
    cpdef unsigned int get_variable(self)
    
    cpdef unsigned int get_knot(self)
    
    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = ?)

cdef class LinearBasisFunction(BasisFunction):
    cdef unsigned int variable
    cdef str label
    
    cpdef translate(LinearBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse)
            
    cpdef FLOAT_t scale(LinearBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts)
    
    cpdef unsigned int get_variable(self)
    
    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = ?)



cdef class Basis:
    '''A wrapper that provides functionality related to a set of BasisFunctions with a 
    common ConstantBasisFunction ancestor.  Retains the order in which BasisFunctions are 
    added.'''
    
    cdef list order
    
    cpdef translate(Basis self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts)
        
    cpdef scale(Basis self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, np.ndarray[FLOAT_t,ndim=1] beta)
    
    cpdef BasisFunction get_root(Basis self)
    
    cpdef append(Basis self, BasisFunction basis_function)
    
    cpdef unsigned int plen(Basis self)
        
    cpdef BasisFunction get(Basis self, unsigned int i)
    
    cpdef transform(Basis self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=2] B)
        
        
        
        