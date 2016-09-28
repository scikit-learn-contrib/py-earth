from cpython cimport bool
cimport numpy as cnp
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

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

    cpdef list get_children(BasisFunction self)
    
    cpdef BasisFunction get_coverage(BasisFunction self, INDEX_t variable)
    
    cpdef bool has_linear(BasisFunction self, INDEX_t variable)
    
    cpdef bool linear_in(BasisFunction self, INDEX_t variable)
    
    cpdef _set_parent(BasisFunction self, BasisFunction parent)

    cpdef _add_child(BasisFunction self, BasisFunction child)

    cpdef BasisFunction get_parent(BasisFunction self)

    cpdef prune(BasisFunction self)

    cpdef unprune(BasisFunction self)

    cpdef knots(BasisFunction self, INDEX_t variable)

    cpdef INDEX_t effective_degree(BasisFunction self)

    cpdef apply(BasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X,
                cnp.ndarray[BOOL_t, ndim=2] missing,
                cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse= ?)

    cpdef cnp.ndarray[INT_t, ndim = 1] valid_knots(BasisFunction self,
        cnp.ndarray[FLOAT_t, ndim=1] values,
        cnp.ndarray[FLOAT_t, ndim=1] variable,
        int variable_idx, INDEX_t check_every,
        int endspan, int minspan,
        FLOAT_t minspan_alpha, INDEX_t n,
        cnp.ndarray[INT_t, ndim=1] workspace)

cdef class RootBasisFunction(BasisFunction):
    cpdef bint covered(RootBasisFunction self, INDEX_t variable)
    
    cpdef bint eligible(RootBasisFunction self, INDEX_t variable)
    
    cpdef set variables(RootBasisFunction self)

    cpdef _smoothed_version(RootBasisFunction self, BasisFunction parent,
                            dict knot_dict, dict translation)

    cpdef INDEX_t degree(RootBasisFunction self)
    
    cpdef _effective_degree(RootBasisFunction self, dict data_dict, dict missing_dict)
    
    cpdef _set_parent(RootBasisFunction self, BasisFunction parent)

    cpdef BasisFunction get_parent(RootBasisFunction self)

    cpdef apply(RootBasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X,
                cnp.ndarray[BOOL_t, ndim=2] missing,
                cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse=?)

    cpdef apply_deriv(RootBasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X,
                      cnp.ndarray[BOOL_t, ndim=2] missing,
                      cnp.ndarray[FLOAT_t, ndim=1] b,
                      cnp.ndarray[FLOAT_t, ndim=1] j, INDEX_t var)

cdef class ConstantBasisFunction(RootBasisFunction):

    cpdef inline FLOAT_t eval(ConstantBasisFunction self)

    cpdef inline FLOAT_t eval_deriv(ConstantBasisFunction self)

cdef class VariableBasisFunction(BasisFunction):
    cdef INDEX_t variable
    cdef readonly label
    
    cpdef INDEX_t degree(VariableBasisFunction self)
    
    cpdef set variables(VariableBasisFunction self)

    cpdef INDEX_t get_variable(VariableBasisFunction self)

cdef class DataVariableBasisFunction(VariableBasisFunction):
    cpdef _effective_degree(DataVariableBasisFunction self, dict data_dict, dict missing_dict)
    
    cpdef bint covered(DataVariableBasisFunction self, INDEX_t variable)
    
    cpdef bint eligible(DataVariableBasisFunction self, INDEX_t variable)
    
    cpdef apply(DataVariableBasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X,
                cnp.ndarray[BOOL_t, ndim=2] missing,
                cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse=?)

    cpdef apply_deriv(DataVariableBasisFunction self,
                      cnp.ndarray[FLOAT_t, ndim=2] X,
                      cnp.ndarray[BOOL_t, ndim=2] missing,
                      cnp.ndarray[FLOAT_t, ndim=1] b,
                      cnp.ndarray[FLOAT_t, ndim=1] j, INDEX_t var)

cdef class MissingnessBasisFunction(VariableBasisFunction):
    cdef readonly bint complement
    
    cpdef _effective_degree(MissingnessBasisFunction self, dict data_dict, dict missing_dict)
    
    cpdef bint covered(MissingnessBasisFunction self, INDEX_t variable)
    
    cpdef bint eligible(MissingnessBasisFunction self, INDEX_t variable)
    
    cpdef bint covered(MissingnessBasisFunction self, INDEX_t variable)
    
    cpdef bint eligible(MissingnessBasisFunction self, INDEX_t variable)
    
    cpdef apply(MissingnessBasisFunction self, cnp.ndarray[FLOAT_t, ndim=2] X,
                cnp.ndarray[BOOL_t, ndim=2] missing,
                cnp.ndarray[FLOAT_t, ndim=1] b, bint recurse=?)

    cpdef apply_deriv(MissingnessBasisFunction self,
                      cnp.ndarray[FLOAT_t, ndim=2] X,
                      cnp.ndarray[BOOL_t, ndim=2] missing,
                      cnp.ndarray[FLOAT_t, ndim=1] b,
                      cnp.ndarray[FLOAT_t, ndim=1] j, INDEX_t var)
    
    cpdef _smoothed_version(MissingnessBasisFunction self, BasisFunction parent,
                            dict knot_dict, dict translation)

cdef class HingeBasisFunctionBase(DataVariableBasisFunction):
    cdef FLOAT_t knot
    cdef INDEX_t knot_idx
    cdef bint reverse

    cpdef bint has_knot(HingeBasisFunctionBase self)

    cpdef INDEX_t get_variable(HingeBasisFunctionBase self)

    cpdef FLOAT_t get_knot(HingeBasisFunctionBase self)

    cpdef bint get_reverse(HingeBasisFunctionBase self)

    cpdef INDEX_t get_knot_idx(HingeBasisFunctionBase self)

cdef class SmoothedHingeBasisFunction(HingeBasisFunctionBase):
    cdef FLOAT_t p
    cdef FLOAT_t r
    cdef FLOAT_t knot_minus
    cdef FLOAT_t knot_plus

    cpdef _smoothed_version(SmoothedHingeBasisFunction self,
                            BasisFunction parent, dict knot_dict,
                            dict translation)

    cpdef get_knot_minus(SmoothedHingeBasisFunction self)

    cpdef get_knot_plus(SmoothedHingeBasisFunction self)

    cpdef _init_p_r(SmoothedHingeBasisFunction self)

    cpdef get_p(SmoothedHingeBasisFunction self)

    cpdef get_r(SmoothedHingeBasisFunction self)

cdef class HingeBasisFunction(HingeBasisFunctionBase):

    cpdef _smoothed_version(HingeBasisFunction self,
                            BasisFunction parent,
                            dict knot_dict, dict translation)

cdef class LinearBasisFunction(DataVariableBasisFunction):
    cpdef bool linear_in(LinearBasisFunction self, INDEX_t variable)
    
    cpdef _smoothed_version(LinearBasisFunction self, BasisFunction parent,
                            dict knot_dict, dict translation)

cdef class Basis:
    '''A wrapper that provides functionality related to a set of BasisFunctions
    with a common RootBasisFunction ancestor.  Retains the order in which
    BasisFunctions are added.'''

    cdef list order
    cdef readonly INDEX_t num_variables
#     cdef dict coverage
    
#     cpdef add_coverage(Basis self, int variable, MissingnessBasisFunction b1, \
#                        MissingnessBasisFunction b2)
#         
#     cpdef get_coverage(Basis self, int variable)
#     
#     cpdef bint has_coverage(Basis self, int variable)

    cpdef int get_num_variables(Basis self)

    cpdef dict anova_decomp(Basis self)

    cpdef smooth(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X)

    cpdef append(Basis self, BasisFunction basis_function)

    cpdef INDEX_t plen(Basis self)

    cpdef BasisFunction get(Basis self, INDEX_t i)

    cpdef transform(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X,
                    cnp.ndarray[BOOL_t, ndim=2] missing,
                    cnp.ndarray[FLOAT_t, ndim=2] B)

    cpdef weighted_transform(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X,
                             cnp.ndarray[BOOL_t, ndim=2] missing,
                             cnp.ndarray[FLOAT_t, ndim=2] B,
                             cnp.ndarray[FLOAT_t, ndim=1] weights)

    cpdef transform_deriv(Basis self, cnp.ndarray[FLOAT_t, ndim=2] X,
                          cnp.ndarray[BOOL_t, ndim=2] missing,
                          cnp.ndarray[FLOAT_t, ndim=1] b,
                          cnp.ndarray[FLOAT_t, ndim=1] j,
                          cnp.ndarray[FLOAT_t, ndim=2] coef,
                          cnp.ndarray[FLOAT_t, ndim=3] J,
                          list variables_of_interest, bool prezeroed_j=?)
