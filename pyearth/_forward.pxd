cimport numpy as cnp
import numpy as np
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t
from _basis cimport Basis
from _record cimport ForwardPassRecord
from _knot_search cimport MultipleOutcomeDependentData

# cdef dict stopping_conditions

cdef class ForwardPasser:

    # User selected parameters
    cdef int endspan
    cdef int minspan
    cdef FLOAT_t endspan_alpha
    cdef FLOAT_t minspan_alpha
    cdef int max_terms
    cdef bint allow_linear
    cdef int max_degree
    cdef FLOAT_t thresh
    cdef FLOAT_t penalty
    cdef int check_every
    cdef int min_search_points
    cdef list xlabels
    cdef FLOAT_t zero_tol
    cdef list fast_heap
    cdef int use_fast
    cdef long fast_K
    cdef long fast_h
    cdef bint allow_missing
    cdef int verbose

    # Input data
    cdef cnp.ndarray X
    cdef cnp.ndarray missing
    cdef cnp.ndarray y
    cdef cnp.ndarray y_col_sum
    cdef cnp.ndarray y_row_sum
    cdef cnp.ndarray sample_weight
    cdef cnp.ndarray output_weight
    cdef INDEX_t m
    cdef INDEX_t n
    cdef FLOAT_t sst
    cdef FLOAT_t y_squared
    cdef FLOAT_t total_weight
    
    # Knot search data
    cdef MultipleOutcomeDependentData outcome
    cdef list predictors
    cdef list workings
    cdef INDEX_t n_outcomes
    
    # Working floating point data
    cdef cnp.ndarray B  # Data matrix in basis space
    cdef cnp.ndarray B_orth  # Orthogonalized version of B
    cdef cnp.ndarray c
    cdef cnp.ndarray c_sqr
    cdef cnp.ndarray norms
    cdef cnp.ndarray u
    cdef cnp.ndarray B_orth_times_parent_cum
    cdef FLOAT_t c_squared
    
    # Working integer data
    cdef cnp.ndarray sort_tracker
    cdef cnp.ndarray sorting
    cdef cnp.ndarray mwork
    cdef cnp.ndarray linear_variables
    cdef int iteration_number
    cdef cnp.ndarray has_missing
    
    # Object construction
    cdef ForwardPassRecord record
    cdef Basis basis

    cpdef Basis get_basis(ForwardPasser self)

    cpdef init_linear_variables(ForwardPasser self)

    cpdef run(ForwardPasser self)

    cdef stop_check(ForwardPasser self)

    cpdef orthonormal_update(ForwardPasser self, b)

    cpdef orthonormal_downdate(ForwardPasser self)

    cdef next_pair(ForwardPasser self)

#     cdef best_knot(ForwardPasser self, INDEX_t parent, cnp.ndarray[FLOAT_t, ndim=1] x,
#                    INDEX_t k, cnp.ndarray[INT_t, ndim=1] candidates,
#                    cnp.ndarray[INT_t, ndim=1] order,
#                    FLOAT_t * mse, FLOAT_t * knot,
#                    INDEX_t * knot_idx)
