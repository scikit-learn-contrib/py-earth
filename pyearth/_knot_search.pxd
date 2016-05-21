cimport cython
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t
from _basis cimport BasisFunction
from _qr cimport UpdatingQT

@cython.final
cdef class SingleWeightDependentData:
    cdef readonly UpdatingQT updating_qt
    cdef readonly FLOAT_t[:] w
    cdef readonly INDEX_t m
    cdef readonly INDEX_t k
    cdef readonly INDEX_t max_terms
    cdef readonly FLOAT_t[:, :] Q_t
    cdef readonly FLOAT_t total_weight
#     cpdef int update_from_basis_function(SingleWeightDependentData self, BasisFunction bf, FLOAT_t[:,:] X, 
#                                          BOOL_t[:,:] missing) except *
    cpdef int update_from_array(SingleWeightDependentData self, FLOAT_t[:] b) except *
#     cpdef int _update(SingleWeightDependentData self, FLOAT_t zero_tol)
    cpdef downdate(SingleWeightDependentData self)
    cpdef reweight(SingleWeightDependentData self, FLOAT_t[:] w, FLOAT_t[:,:] B, INDEX_t k)
    
@cython.final
cdef class MultipleOutcomeDependentData:
    cdef list outcomes
    cdef list weights
    cpdef update_from_array(MultipleOutcomeDependentData self, FLOAT_t[:] b)
    cpdef downdate(MultipleOutcomeDependentData self)
    cpdef list sse(MultipleOutcomeDependentData self)
    cpdef FLOAT_t mse(MultipleOutcomeDependentData self)
    
@cython.final
cdef class SingleOutcomeDependentData:
    cdef readonly FLOAT_t[:] y
    cdef readonly SingleWeightDependentData weight
    cdef readonly FLOAT_t[:] theta
    cdef public FLOAT_t omega
    cdef public FLOAT_t sse_
    cdef public INDEX_t m
    cdef public INDEX_t k
    cdef public INDEX_t max_terms
    cdef public object householder
    cpdef FLOAT_t sse(SingleOutcomeDependentData self)
    cpdef int synchronize(SingleOutcomeDependentData self) except *
    cpdef int update(SingleOutcomeDependentData self) except *
    cpdef downdate(SingleOutcomeDependentData self)


@cython.final
cdef class PredictorDependentData:
    cdef readonly FLOAT_t[:] p
    cdef readonly FLOAT_t[:] x
    cdef readonly FLOAT_t[:] candidates
    cdef readonly INDEX_t[:] order

@cython.final
cdef class KnotSearchReadOnlyData:
    cdef readonly PredictorDependentData predictor
    cdef readonly MultipleOutcomeDependentData outcome

@cython.final
cdef class KnotSearchState:
    cdef public FLOAT_t alpha
    cdef public FLOAT_t beta
    cdef public FLOAT_t lambda_
    cdef public FLOAT_t mu
    cdef public FLOAT_t upsilon
    cdef public FLOAT_t phi
    cdef public FLOAT_t phi_next
    cdef public INDEX_t ord_idx
    cdef public INDEX_t idx
    cdef public FLOAT_t zeta_squared

@cython.final
cdef class KnotSearchWorkingData:
    cdef readonly FLOAT_t[:] gamma
    cdef readonly FLOAT_t[:] kappa
    cdef readonly FLOAT_t[:] delta_kappa
    cdef readonly FLOAT_t[:] chi
    cdef readonly FLOAT_t[:] psi
    cdef KnotSearchState state

@cython.final
cdef class KnotSearchData:
    cdef readonly KnotSearchReadOnlyData constant
    cdef readonly list workings
    cdef public INDEX_t q

cdef dot(FLOAT_t[:] x1, FLOAT_t[:] x2, INDEX_t q)
cdef w2dot(FLOAT_t[:] w, FLOAT_t[:] x1, FLOAT_t[:] x2, INDEX_t q)
cdef wdot(FLOAT_t[:] w, FLOAT_t[:] x1, FLOAT_t[:] x2, INDEX_t q)
cdef inline void fast_update(PredictorDependentData predictor, SingleOutcomeDependentData outcome, 
                        KnotSearchWorkingData working, FLOAT_t[:] p, INDEX_t q, INDEX_t m ,INDEX_t r) except *
cpdef tuple knot_search(KnotSearchData data, FLOAT_t[:] candidates, FLOAT_t[:] p, INDEX_t q, INDEX_t m, INDEX_t r, INDEX_t n_outcomes, int verbose)

