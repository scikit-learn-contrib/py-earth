cimport cython
from _types cimport FLOAT_t, INT_t, INDEX_t, BOOL_t

@cython.final
cdef class OutcomeDependentData:
    cdef readonly FLOAT_t[:,:] Q_t
    cdef readonly FLOAT_t[:] y
    cdef readonly FLOAT_t[:] w
    cdef readonly FLOAT_t[:] theta
    cdef public FLOAT_t omega
    cdef public INDEX_t m
    cdef public INDEX_t k
    cdef public INDEX_t max_terms
    cpdef int update(OutcomeDependentData self, FLOAT_t[:] b, FLOAT_t zero_tol) except *
    cpdef downdate(OutcomeDependentData self)
    cpdef reweight(OutcomeDependentData self, FLOAT_t[:] w, FLOAT_t[:,:] B, INDEX_t k, FLOAT_t zero_tol)
    
@cython.final
cdef class PredictorDependentData:
    cdef readonly FLOAT_t[:] p
    cdef readonly FLOAT_t[:] x
    cdef readonly FLOAT_t[:] candidates
    cdef readonly INT_t[:] order

@cython.final
cdef class KnotSearchReadOnlyData:
    cdef readonly PredictorDependentData predictor
    cdef readonly list outcomes

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
cdef void fast_update(PredictorDependentData predictor, OutcomeDependentData outcome, 
                        KnotSearchWorkingData working, FLOAT_t[:] p, INDEX_t q, INDEX_t m ,INDEX_t r) except *
cpdef tuple knot_search(KnotSearchData data, FLOAT_t[:] candidates, FLOAT_t[:] p, INDEX_t q, INDEX_t m, INDEX_t r, INDEX_t n_outcomes)

