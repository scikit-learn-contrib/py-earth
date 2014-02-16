cimport numpy as cnp
ctypedef cnp.float64_t FLOAT_t
ctypedef cnp.intp_t INT_t
ctypedef cnp.ulong_t INDEX_t
ctypedef cnp.uint8_t BOOL_t

cdef class BasisFunction:
    '''Abstract.  Subclasses must implement the apply and __init__ methods.'''

    cdef BasisFunction parent
    cdef dict child_map
    cdef list children
    cdef bint pruned
    cdef bint prunable
    cdef bint splittable
    
    cpdef smooth(BasisFunction self, dict knot_dict, dict translation)
    
    cpdef bint has_knot(BasisFunction self)

    cpdef bint is_prunable(BasisFunction self)

    cpdef bint is_pruned(BasisFunction self)

    cpdef bint is_splittable(BasisFunction self)

    cpdef bint make_splittable(BasisFunction self)

    cpdef bint make_unsplittable(BasisFunction self)

    cdef list get_children(BasisFunction self)

    cpdef _set_parent(BasisFunction self, BasisFunction parent)

    cpdef _add_child(BasisFunction self, BasisFunction child)

    cpdef BasisFunction get_parent(BasisFunction self)

    cpdef prune(BasisFunction self)

    cpdef unprune(BasisFunction self)

    cpdef knots(BasisFunction self, INDEX_t variable)

    cpdef INDEX_t degree(BasisFunction self)

    cpdef apply(BasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse= ?)

    cpdef cnp.ndarray[INT_t, ndim = 1] valid_knots(BasisFunction self, cnp.ndarray[FLOAT_t, ndim=1] values, cnp.ndarray[FLOAT_t, ndim=1] variable, int variable_idx, INDEX_t check_every, int endspan, int minspan, FLOAT_t minspan_alpha, INDEX_t n, cnp.ndarray[INT_t, ndim=1] workspace)

cdef class RootBasisFunction(BasisFunction):

    cpdef set variables(RootBasisFunction self)
    
    cpdef _smoothed_version(RootBasisFunction self, BasisFunction parent, dict knot_dict, dict translation)
        
    cpdef INDEX_t degree(RootBasisFunction self)
    
    cpdef _set_parent(RootBasisFunction self, BasisFunction parent)

    cpdef BasisFunction get_parent(RootBasisFunction self)
        
cdef class ConstantBasisFunction(RootBasisFunction):

    cpdef apply(self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse= ?)

cdef class ZeroBasisFunction(RootBasisFunction):
    cpdef apply(self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse=?)


cdef class HingeBasisFunctionBase(BasisFunction):
    cdef FLOAT_t knot
    cdef INDEX_t knot_idx
    cdef INDEX_t variable
    cdef bint reverse
    cdef str label
    
    cpdef bint has_knot(HingeBasisFunctionBase self)
        
    cpdef INDEX_t get_variable(HingeBasisFunctionBase self)
    
    cpdef FLOAT_t get_knot(HingeBasisFunctionBase self)
    
    cpdef bint get_reverse(HingeBasisFunctionBase self)
    
    cpdef INDEX_t get_knot_idx(HingeBasisFunctionBase self)
    
    cpdef set variables(HingeBasisFunctionBase self)

cdef class SmoothedHingeBasisFunction(HingeBasisFunctionBase):
    cdef FLOAT_t p
    cdef FLOAT_t r
    
    cpdef _smoothed_version(SmoothedHingeBasisFunction self, BasisFunction parent, dict knot_dict, dict translation)
    
    cpdef _init_p_r(SmoothedHingeBasisFunction self)
    
    cpdef apply(self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse=?)
    
cdef class HingeBasisFunction(HingeBasisFunctionBase):
    
    cpdef _smoothed_version(HingeBasisFunction self, BasisFunction parent, dict knot_dict, dict translation)
    
    cpdef apply(self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse= ?)

cdef class LinearBasisFunction(BasisFunction):
    cdef INDEX_t variable
    cdef str label
    
    cpdef _smoothed_version(LinearBasisFunction self, BasisFunction parent, dict knot_dict, dict translation)

    cpdef INDEX_t get_variable(self)

    cpdef apply(self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse= ?)

cdef class Basis:
    '''A wrapper that provides functionality related to a set of BasisFunctions with a
    common RootBasisFunction ancestor.  Retains the order in which BasisFunctions are
    added.'''

    cdef list order
    cdef readonly INDEX_t num_variables
    
    cpdef dict anova_decomp(Basis self)
    
    cpdef smooth(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X)
	
    cpdef append(Basis self, BasisFunction basis_function)

    cpdef INDEX_t plen(Basis self)

    cpdef BasisFunction get(Basis self, INDEX_t i)

    cpdef transform(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=2] B)

    cpdef weighted_transform(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=2] B, cnp.ndarray[FLOAT_t, ndim=1] weights)
