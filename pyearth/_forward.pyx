# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False

from ._util cimport gcv_adjust, log2, apply_weights_1d, apply_weights_slice
from ._basis cimport (Basis, BasisFunction, ConstantBasisFunction,
                      HingeBasisFunction, LinearBasisFunction, 
                      MissingnessBasisFunction)
from ._record cimport ForwardPassIteration
from ._types import BOOL

from libc.math cimport sqrt, abs, log
import numpy as np
cnp.import_array()

from heapq import heappush, heappop
class FastHeapContent:

    def __init__(self, idx, mse=-np.inf, m=-np.inf, v=None):
        """
        This class defines an entry of the priority queue as defined in [1].
        The entry stores information about parent basis functions and is
        used by the priority queue in the forward pass 
        to choose the next parent basis function to try.

        References
        ----------
        .. [1] Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993. 

        """
        self.idx = idx
        self.mse = mse
        self.m = m
        self.v = v

    def __lt__(self, other):
        return self.mse < other.mse

stopping_conditions = {
    MAXTERMS: "Reached maximum number of terms",
    MAXRSQ: "Achieved RSQ value within threshold of 1",
    NOIMPRV: "Improvement below threshold",
    LOWGRSQ: "GRSQ too low",
    NOCAND: "No remaining candidate knot locations"
}

cdef class ForwardPasser:

    def __init__(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X,
                 cnp.ndarray[BOOL_t, ndim=2] missing,
                 cnp.ndarray[FLOAT_t, ndim=2] y,
                 cnp.ndarray[FLOAT_t, ndim=1] sample_weight,
                 cnp.ndarray[FLOAT_t, ndim=1] output_weight,
                 **kwargs):
        cdef INDEX_t i
        self.X = X
        self.missing = missing
        self.y = y * np.sqrt(sample_weight[:, np.newaxis])
        self.sample_weight = sample_weight
        self.output_weight = output_weight
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan       = kwargs.get('endspan', -1)
        self.minspan       = kwargs.get('minspan', -1)
        self.endspan_alpha = kwargs.get('endspan_alpha', .05)
        self.minspan_alpha = kwargs.get('minspan_alpha', .05)
        self.max_terms     = kwargs.get('max_terms', 2 * self.n + 10)
        self.allow_linear  = kwargs.get('allow_linear', True)
        self.max_degree    = kwargs.get('max_degree', 1)
        self.thresh        = kwargs.get('thresh', 0.001)
        self.penalty       = kwargs.get('penalty', 3.0)
        self.check_every   = kwargs.get('check_every', -1)
        self.min_search_points = kwargs.get('min_search_points', 100)
        self.xlabels       = kwargs.get('xlabels')
        self.use_fast = kwargs.get('use_fast', False)
        self.fast_K = kwargs.get("fast_K", 5)
        self.fast_h = kwargs.get("fast_h", 1)
        self.zero_tol = kwargs.get('zero_tol', 1e-12)
        self.allow_missing = kwargs.get("allow_missing", False)
        if self.allow_missing:
            self.has_missing = np.any(self.missing, axis=0).astype(BOOL)

        self.fast_heap = []

        if self.xlabels is None:
            self.xlabels = ['x' + str(i) for i in range(self.n)]
        if self.check_every < 0:
            self.check_every = (<int > (self.m / self.min_search_points)
                                if self.m > self.min_search_points
                                else 1)

        self.y_squared = ((self.y** 2) * self.output_weight).sum()
        stuff_per_example = ((((np.sqrt(self.sample_weight[:, np.newaxis]) * y)).sum(axis=0) ** 2) / self.sample_weight.sum())
        stuff = (stuff_per_example * self.output_weight).sum()
        self.sst = (self.y_squared - stuff) / (self.m)

        self.record = ForwardPassRecord(
            self.m, self.n, self.penalty, self.sst, self.xlabels)
        self.basis = Basis(self.n)
        self.basis.append(ConstantBasisFunction())
        if self.use_fast is True:
            heappush(self.fast_heap, FastHeapContent(idx=0))

        self.sorting = np.empty(shape=self.m, dtype=np.int)
        self.mwork = np.empty(shape=self.m, dtype=np.int)
        self.u = np.empty(shape=self.max_terms, dtype=float)
        self.B_orth_times_parent_cum = np.empty(
            shape=self.max_terms, order='F', dtype=np.float)
        self.B = np.ones(
            shape=(self.m, self.max_terms), order='F', dtype=np.float)
        self.basis.weighted_transform(self.X, self.missing, self.B[:,0:1], self.sample_weight)
        # An orthogonal matrix with the same column space as B
        self.B_orth = self.B.copy()
        self.u = np.empty(shape=self.max_terms, dtype=np.float)
        self.c = np.empty(shape=(self.max_terms, self.y.shape[1]),
                          dtype=np.float)
        self.norms = np.empty(shape=self.max_terms, dtype=np.float)
        self.c_squared = 0.0
        self.sort_tracker = np.empty(shape=self.m, dtype=np.int)
        for i in range(self.m):
            self.sort_tracker[i] = i

        self.linear_variables = np.zeros(shape=self.n, dtype=np.int)
        self.init_linear_variables()

        self.iteration_number = 0

        # Add in user selected linear variables
        for linvar in kwargs.get('linvars',[]):
            if linvar in self.xlabels:
                self.linear_variables[self.xlabels.index(linvar)] = 1
            elif linvar in range(self.n):
                self.linear_variables[linvar] = 1
            else:
                raise IndexError(
                    'Unknown variable selected in linvars argument.')

        # Initialize B_orth, c, and c_squared (assuming column 0 of B_orth is
        # already filled with 1)
        self.orthonormal_update(0)

    cpdef Basis get_basis(ForwardPasser self):
        return self.basis

    cpdef init_linear_variables(ForwardPasser self):
        cdef INDEX_t variable
        cdef INDEX_t endspan
        cdef cnp.ndarray[INT_t, ndim = 1] order
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha / self.n))
        else:
            endspan = self.endspan
        cdef ConstantBasisFunction root_basis_function = self.basis[0]
        for variable in range(self.n):
            order = np.argsort(X[:, variable])[::-1].astype(np.int)
            if root_basis_function.valid_knots(B[order, 0], X[order, variable],
                                               variable, self.check_every,
                                               endspan, self.minspan,
                                               self.minspan_alpha, self.n,
                                               self.mwork).shape[0] == 0:
                linear_variables[variable] = 1
            else:
                linear_variables[variable] = 0

    def get_B_orth(ForwardPasser self):
        return self.B_orth

    cpdef run(ForwardPasser self):
        if self.max_terms > 1:
            while True:
                self.next_pair()
                if self.stop_check():
                    break
                self.iteration_number += 1

    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1
        if self.record.iterations[last].get_size() + 4 > self.max_terms:
            self.record.stopping_condition = MAXTERMS
            return True
        rsq = self.record.rsq(last)
        if rsq > 1 - self.thresh:
            self.record.stopping_condition = MAXRSQ
            return True
        if last > 0:
            previous_rsq = self.record.rsq(last - 1)
            if rsq - previous_rsq < self.thresh:
                self.record.stopping_condition = NOIMPRV
                return True
        if self.record.grsq(last) < -10:
            self.record.stopping_condition = LOWGRSQ
            return True
        if self.record.iterations[last].no_further_candidates():
            self.record.stopping_condition = NOCAND
            return True
        return False

    cpdef int orthonormal_update(ForwardPasser self, INDEX_t k):
        '''Orthogonalize and normalize column k of B_orth against all previous
           columns of B_orth.'''
        # Currently implemented using modified Gram-Schmidt process

        cdef cnp.ndarray[FLOAT_t, ndim = 2] B_orth = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B_orth)

        cdef INDEX_t i
        cdef FLOAT_t nrm
        cdef FLOAT_t nrm0

        # Get the original norm
        nrm0 = sqrt(np.dot(B_orth[:, k], B_orth[:, k]))

        # Orthogonalize
        for i in range(k):
            B_orth[:, k] -= B_orth[:, i] * np.dot(B_orth[:, k], B_orth[:, i])

        # Normalize
        self.norms[k] = nrm = sqrt(np.dot(B_orth[:, k], B_orth[:, k]))

        if nrm0 <= self.zero_tol or nrm / nrm0 <= self.zero_tol:
            B_orth[:, k] = 0.0
            for p in range(self.y.shape[1]):
                self.c[k, p] = 0.0
            # The new column is in the column space of the previous columns
            return 1
        B_orth[:, k] /= nrm

        # Update c
        self.c[k] = (B_orth[:, k][:, np.newaxis] * self.y).sum(axis=0)
        self.c_squared += ( (self.c[k] ** 2) * self.output_weight).sum()

        return 0  # No problems

    cpdef orthonormal_downdate(ForwardPasser self, INDEX_t k):
        '''
        Undo the effects of the last orthonormal update.  You can only undo the
        last orthonormal update this way. There will be no warning of any kind
        if you mess this up.  You'll just get wrong answers. In reality, all
        this does is downdate c_squared (the elements of c and B_orth are left
        alone, since they can simply be ignored until they are overwritten).
        '''
        self.c_squared -= ( (self.c[k]**2) * self.output_weight).sum()

    def trace(self):
        return self.record

    cdef next_pair(ForwardPasser self):
        cdef INDEX_t variable
        cdef INDEX_t parent_idx
        cdef INDEX_t parent_degree
        cdef INDEX_t nonzero_count
        cdef BasisFunction parent
        cdef cnp.ndarray[INT_t, ndim = 1] candidates_idx
        cdef FLOAT_t knot
        cdef FLOAT_t mse
        cdef INDEX_t knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef FLOAT_t mse_choice_cur_parent
        cdef int variable_choice_cur_parent
        cdef int knot_idx_choice
        cdef INDEX_t parent_idx_choice
        cdef BasisFunction parent_choice
        parent_basis_content_choice = None
        parent_basis_content = None
        cdef INDEX_t variable_choice
        cdef bint first = True
        cdef BasisFunction bf1
        cdef BasisFunction bf2
        cdef BasisFunction bf3
        cdef BasisFunction bf4
        cdef bint already_covered
        cdef INDEX_t k = len(self.basis)
        cdef INDEX_t endspan
        cdef bint linear_dependence
        cdef bint dependent
        cdef FLOAT_t gcv_factor_k_plus_1 = gcv_adjust(k + 1, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_2 = gcv_adjust(k + 2, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_
        cdef FLOAT_t mse_
        cdef INDEX_t i
        cdef bint eligible
        cdef bint covered
        cdef bint missing_flag
        cdef bint choice_needs_coverage
        cdef int max_variable_degree
        
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[BOOL_t, ndim = 2] missing = (
            <cnp.ndarray[BOOL_t, ndim = 2] > self.missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B_orth = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B_orth)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] y = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.y)
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[INT_t, ndim = 1] sorting = (
            <cnp.ndarray[INT_t, ndim = 1] > self.sorting)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] sample_weight = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.sample_weight)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] output_weight = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.output_weight)
        cdef cnp.ndarray[BOOL_t, ndim = 1] has_missing = (
            <cnp.ndarray[BOOL_t, ndim = 1] > self.has_missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] x

        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha / self.n))
        else:
            endspan = self.endspan

        if self.use_fast is True:
            # choose only among the top "fast_K" basis functions
            # as parents
            nb_basis = min(self.fast_K, k)
        else:
            nb_basis = k
        
        content_to_be_repushed = []
        for idx in range(nb_basis):
            # Iterate over parents
            if self.use_fast is True:
                # retrieve the next basis function to try as parent
                parent_basis_content = heappop(self.fast_heap)
                content_to_be_repushed.append(parent_basis_content)
                parent_idx = parent_basis_content.idx
                mse_choice_cur_parent = -1
                variable_choice_cur_parent = -1
            else:
                parent_idx = idx

            parent = self.basis.get(parent_idx)
            
            if not parent.is_splittable():
                continue

            if self.use_fast is True:
                # each "fast_h" iteration, force to pass through all the variables,
                if self.iteration_number - parent_basis_content.m > self.fast_h:
                    variables = range(self.n)
                    parent_basis_content.m = self.iteration_number
                # in the opposite case, just use the last chosen variable
                else:
                    variables = [parent_basis_content.v]
                variables = range(self.n)
            else:
                variables = range(self.n)
            
            parent_degree = parent.degree()
            for variable in variables:
                
                # Determine whether missingness needs to be accounted for.
                if self.allow_missing and has_missing[variable]:
                    missing_flag = True
                    eligible = parent.eligible(variable)
                    covered = parent.covered(variable)
                    max_variable_degree = self.max_degree + 1
                else:
                    missing_flag = False
                    max_variable_degree = self.max_degree
                
                # Make sure not to exceed max_degree (but don't count the 
                # covering missingness basis function if required)
                if self.max_degree >= 0:
                    if parent_degree >= max_variable_degree:
                        continue
                
                # If there is missing data and this parent is not 
                # an eligible parent for this variable with missingness
                # (because it includes a non-missing factor for the variable)
                # the skip this variable.
                if missing_flag and not eligible:
                    continue
                
                # If necessary, protect from missing data
                if missing_flag:
                    x = X[:, variable].copy()
                    x[missing[:, variable]==1] = 0.0
                else:
                    x = X[:, variable]

                # Sort the data
                # TODO: eliminate Python call / data copy
                sorting[:] = np.argsort(x)[::-1]

                linear_dependence = False


                # Add the linear term to B
                for i in range(self.m):
                    B[i, k] = B[i, parent_idx] * x[i]

                # Orthonormalize
                for i in range(self.m):
                    B_orth[i, k] = B[i, k]
                linear_dependence = self.orthonormal_update(k)
                if missing_flag and not covered:
                    for i in range(self.m):
                        B_orth[i, k + 1] = B[i, parent_idx] * (1 - missing[i, variable])
                    self.orthonormal_update(k + 1)
                    for i in range(self.m):
                        B_orth[i, k + 2] = B[i, parent_idx] * missing[i, variable]
                    self.orthonormal_update(k + 2)
                

                # If a new hinge function does not improve the gcv over the
                # linear term then just the linear term will be retained
                # (if allow_linear).  Calculate the gcv with just the linear
                # term in order to compare later.  Note that the mse with
                # another term never increases, but the gcv may because it
                # penalizes additional terms.
                mse_ = (self.y_squared - self.c_squared) / self.m
                gcv_ = gcv_factor_k_plus_1 * \
                    (self.y_squared - self.c_squared) / self.m

                if linear_variables[variable]:
                    mse = mse_
                    knot_idx = -1
                else:

                    # Find the valid knot candidates
                    candidates_idx = parent.valid_knots(B[sorting, parent_idx],
                                                        x[sorting],
                                                        variable,
                                                        self.check_every,
                                                        endspan, self.minspan,
                                                        self.minspan_alpha,
                                                        self.n, self.mwork)

                    if len(candidates_idx) > 0:
                    # Choose the best candidate (if no candidate is an
                    # improvement on the linear term in terms of gcv, knot_idx
                    # is set to -1

                        # Find the best knot location for this parent and
                        # variable combination
                        if missing_flag and not covered:
                            self.best_knot(parent_idx, x, k + 2, candidates_idx,
                                           sorting, & mse, & knot, & knot_idx)
                        else:
                            self.best_knot(parent_idx, x, k, candidates_idx,
                                           sorting, & mse, & knot, & knot_idx)

                        # If the hinge function does not decrease the gcv then
                        # just keep the linear term (if allow_linear is True)
                        if (self.allow_linear and
                            (gcv_factor_k_plus_2 * mse >= gcv_)):
                            mse = mse_
                            knot_idx = -1

                    else:
                        # Do an orthonormal downdate and skip to the next
                        # iteration
                        if missing_flag and not covered:
                            # Order matters here: orthonormal updates
                            # must be undone in the reverse order in which
                            # they were added.
                            self.orthonormal_downdate(k + 2)
                            self.orthonormal_downdate(k + 1)
                        self.orthonormal_downdate(k)
                        continue

                # Do an orthonormal downdate
                if missing_flag and not covered:
                    # Order matters here: orthonormal updates
                    # must be undone in the reverse order in which
                    # they were added.
                    self.orthonormal_downdate(k + 2)
                    self.orthonormal_downdate(k + 1)
                self.orthonormal_downdate(k)
                
                # Update the choices
                if mse < mse_choice or first:
                    if first:
                        first = False
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    if self.use_fast is True:
                        parent_basis_content_choice = parent_basis_content
                    variable_choice = variable
                    dependent = linear_dependence
                    if missing_flag and not covered:
                        choice_needs_coverage = True
                    else:
                        choice_needs_coverage = False
                
                if self.use_fast is True:
                    if (mse_choice_cur_parent == -1) or (mse < mse_choice_cur_parent):
                        mse_choice_cur_parent = mse
                        variable_choice_cur_parent = variable
            if self.use_fast is True:
                if mse_choice_cur_parent != -1:
                    parent_basis_content.mse = mse_choice_cur_parent
                    parent_basis_content.v = variable_choice_cur_parent
        
        if self.use_fast is True:
            for content in content_to_be_repushed:
                heappush(self.fast_heap, content)

        # Make sure at least one candidate was checked
        if first:
            self.record[len(self.record) - 1].set_no_candidates(True)
            return
        
        # Add the new basis functions
        label = self.xlabels[variable_choice]
        if choice_needs_coverage:
            bf3 = MissingnessBasisFunction(parent_choice, variable_choice,
                                           True, label)
            bf4 = MissingnessBasisFunction(parent_choice, variable_choice,
                                           False, label)
            if self.basis.has_coverage(variable_choice):
                bf3, bf4 = self.basis.get_coverage(variable_choice)
                already_covered = True
            else:
                already_covered = False
            parent_choice = bf3
        if knot_idx_choice != -1:
            # Add the new basis functions
            bf1 = HingeBasisFunction(parent_choice,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     False, label)
            bf2 = HingeBasisFunction(parent_choice,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     True, label)
            
            bf1.apply(X, missing, B[:, k])
            apply_weights_slice(B, sample_weight, k)
            bf2.apply(X, missing, B[:, k + 1])
            apply_weights_slice(B, sample_weight, k + 1)

            self.basis.append(bf1)        
            self.basis.append(bf2)
            
            if choice_needs_coverage:
                if not already_covered:
                    bf3.apply(X, missing, B[:, k + 2])
                    apply_weights_slice(B, sample_weight, k + 2)
                    bf4.apply(X, missing, B[:, k + 3])
                    apply_weights_slice(B, sample_weight, k + 3)
                    self.basis.add_coverage(variable_choice, bf3, bf4) 

            if self.use_fast is True:
                bf1_content = FastHeapContent(idx=k)
                heappush(self.fast_heap, bf1_content)
                bf2_content = FastHeapContent(idx=k + 1)
                heappush(self.fast_heap, bf2_content)
                if choice_needs_coverage:
                    if not already_covered:
                        bf3_content = FastHeapContent(idx=k + 2)
                        heappush(self.fast_heap, FastHeapContent(idx=k + 2))
                        bf4_content = FastHeapContent(idx=k + 3)
                        heappush(self.fast_heap, FastHeapContent(idx=k + 3))
                    
            # Orthogonalize the new basis
            B_orth[:, k] = B[:, k]
            if self.orthonormal_update(k) == 1:
                bf1.make_unsplittable()
            B_orth[:, k + 1] = B[:, k + 1]
            if self.orthonormal_update(k + 1) == 1:
                bf2.make_unsplittable()
            if choice_needs_coverage:
                if not already_covered:
                    B_orth[:, k + 2] = B[:, k + 2]
                    if self.orthonormal_update(k + 2) == 1:
                        pass
#                         bf3.make_unsplittable()
                    B_orth[:, k + 3] = B[:, k + 3]
                    if self.orthonormal_update(k + 3) == 1:
                        pass
#                         bf4.make_unsplittable()
        elif not dependent and knot_idx_choice == -1:
            # In this case, only add the linear basis function (in addition to 
            # covering missingness basis functions if needed)
            if choice_needs_coverage:
                bf2 = MissingnessBasisFunction(parent_choice, variable_choice,
                                               True, label)
                bf3 = MissingnessBasisFunction(parent_choice, variable_choice,
                                               False, label)
                if self.basis.has_coverage(variable_choice):
                    bf2, bf3 = self.basis.get_coverage(variable_choice)
                    already_covered = True
                else:
                    already_covered = False
                parent_choice = bf2
            bf1 = LinearBasisFunction(parent_choice, variable_choice, label)
            bf1.apply(X, missing, B[:, k])
            apply_weights_slice(B, sample_weight, k)
            self.basis.append(bf1)
            if choice_needs_coverage:
                if not already_covered:
                    bf2.apply(X, missing, B[:, k + 1])
                    apply_weights_slice(B, sample_weight, k + 1)
                    bf3.apply(X, missing, B[:, k + 2])
                    apply_weights_slice(B, sample_weight, k + 2)
                    self.basis.add_coverage(variable_choice, bf2, bf3)
            if self.use_fast is True:
                bf1_content = FastHeapContent(idx=k)
                heappush(self.fast_heap, bf1_content)
                if choice_needs_coverage:
                    if not already_covered:
                        bf2_content = FastHeapContent(idx=k + 1)
                        heappush(self.fast_heap, bf2_content)
                        bf3_content = FastHeapContent(idx=k + 2)
                        heappush(self.fast_heap, bf3_content)
            # Orthogonalize the new basis
            B_orth[:, k] = B[:, k]
            if self.orthonormal_update(k) == 1:
                bf1.make_unsplittable()
            if choice_needs_coverage:
                if not already_covered:
                    B_orth[:, k + 1] = B[:, k + 1]
                    if self.orthonormal_update(k + 1) == 1:
                        pass
#                         bf2.make_unsplittable()
                    B_orth[:, k + 2] = B[:, k + 2]
                    if self.orthonormal_update(k + 2) == 1:
                        pass
#                         bf3.make_unsplittable()
        else:  # dependent and knot_idx_choice == -1
            # In this case there were no acceptable choices remaining, so end
            # the forward pass
            self.record[len(self.record) - 1].set_no_candidates(True)
            return
        if self.use_fast is True: 
            parent_basis_content_choice.m = -np.inf

        # Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,
                                                variable_choice,
                                                knot_idx_choice, mse_choice,
                                                len(self.basis)))

    cdef best_knot(ForwardPasser self, INDEX_t parent, cnp.ndarray[FLOAT_t, ndim=1] x,
                   INDEX_t k, cnp.ndarray[INT_t, ndim=1] candidates,
                   cnp.ndarray[INT_t, ndim=1] order,
                   FLOAT_t * mse, FLOAT_t * knot,
                   INDEX_t * knot_idx):
        '''
        Find the best knot location (in terms of squared error).

        Assumes:
        B[:,k] is the linear term for variable
        X[:,variable] is in decreasing order
        candidates is in increasing order (it is an array of indices
        into X[:,variable] mse is a pointer to the mean squared error of
        including just the linear term in B[:,k]
        '''

        cdef cnp.ndarray[FLOAT_t, ndim = 1] b = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.B[:, k + 1])
        cdef cnp.ndarray[FLOAT_t, ndim = 1] b_parent = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.B[:, parent])
        cdef cnp.ndarray[FLOAT_t, ndim = 1] u = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.u)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B_orth = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B_orth)
#         cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
#             <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] y = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.y)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] c = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.c)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] B_orth_times_parent_cum = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.B_orth_times_parent_cum)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] sample_weight = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.sample_weight)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] output_weight = (
            <cnp.ndarray[FLOAT_t, ndim = 1] > self.output_weight)



        cdef INDEX_t num_candidates = candidates.shape[0]

        cdef INDEX_t h
        cdef INDEX_t i
        cdef INDEX_t j
        cdef INDEX_t h_
        cdef INDEX_t i_
        cdef INDEX_t j_
        cdef FLOAT_t u_end
        cdef cnp.ndarray[FLOAT_t, ndim = 1] c_end = np.zeros(self.y.shape[1])
        cdef FLOAT_t z_end_squared
        cdef INDEX_t candidate_idx
        cdef INDEX_t last_candidate_idx
        cdef INDEX_t last_last_candidate_idx
        cdef INDEX_t best_candidate_idx
        cdef FLOAT_t candidate
        cdef FLOAT_t last_candidate
        cdef FLOAT_t best_candidate
        cdef FLOAT_t best_z_end_squared
        cdef FLOAT_t y_cum
        cdef FLOAT_t b_times_parent_cum
        cdef FLOAT_t diff
        cdef FLOAT_t delta_b_squared
        cdef cnp.ndarray[FLOAT_t, ndim = 1] delta_c_end = np.zeros(self.y.shape[1])
        cdef FLOAT_t delta_u_end
        cdef FLOAT_t parent_squared_cum
        cdef cnp.ndarray[FLOAT_t, ndim = 1] parent_times_y_cum = np.zeros(self.y.shape[1])
        cdef cnp.ndarray[FLOAT_t, ndim = 1] u_dot_c = np.zeros(self.y.shape[1])
        cdef FLOAT_t u_dot_u
        cdef FLOAT_t float_tmp
        cdef FLOAT_t delta_b_j
        cdef FLOAT_t z_denom

        # Compute the initial n the R package earth, Stephen Milborrow gets
        # around this problem by only allowing a separable weight matrix. That
        # is, there are row weights and (output) column weights, so the
        # resulting weight matrix is basically an outer product of the two. That
        # way no additional copy of B or B_orth is needed because they would all
        # be simply scalar multiples of each other.
        candidate_idx = candidates[0]
        candidate = x[order[candidate_idx]]
        for i in range(self.m):  # TODO: Vectorize?
            b[i] = 0
        for i_ in range(self.m):
            i = order[i_]
            float_tmp = x[i] - candidate
            if float_tmp > 0:
                b[i] = b_parent[i] * float_tmp
            else:
                break

        # Compute the initial covariance column, u (not including the final
        # element)
        u[0:k + 1] = np.dot(b, B_orth[:, 0:k + 1])

        # Compute the new last elements of c and u
        for p in range(self.y.shape[1]):
            c_end[p] = 0.0
        u_end = 0.0
        for i in range(self.m):
            u_end += b[i] * b[i]
            for p in range(self.y.shape[1]):
                c_end[p] += b[i] * y[i, p]

        # Compute the last element of z (the others are identical to c)
        for p in range(self.y.shape[1]):
            u_dot_c[p] = 0.0
        u_dot_u = 0.0
        for i in range(k + 1):
            u_dot_u += u[i] * u[i]
            for p in range(self.y.shape[1]):
                u_dot_c[p] +=  u[i] * c[i, p]
        z_denom = (u_end - u_dot_u)
        z_end_squared = 0.
        for p in range(self.y.shape[1]):
            z_end_squared += ((c_end[p] - u_dot_c[p]) ** 2) * (output_weight[p])
        if (z_denom <= self.zero_tol) and (z_denom <= (self.zero_tol * z_end_squared)) :
            z_end_squared = np.nan
        else:
            z_end_squared /= z_denom

        # Minimizing the norm is actually equivalent to maximizing z_end_squared
        # Store z_end_squared and the current candidate as the best knot choice
        best_z_end_squared = z_end_squared
        best_candidate_idx = candidate_idx
        best_candidate = candidate

        # Initialize the accumulators
        i = order[0]
        last_candidate_idx = 0
        y_cum = y[i, 0]
        B_orth_times_parent_cum[0:k + 1] = B_orth[i, 0:k + 1] * b_parent[i]
        b_times_parent_cum = b[i] * b_parent[i]
        parent_squared_cum = b_parent[i] ** 2

        for p in range(self.y.shape[1]):
            parent_times_y_cum[p] +=  b_parent[i] * y[i, p]

        # Now loop over the remaining candidates and update z_end_squared for
        # each, looking for the greatest value
        for i_ in range(1, num_candidates):
            i = order[i_]

            # Update the candidate
            last_last_candidate_idx = last_candidate_idx
            last_candidate_idx = candidate_idx
            last_candidate = candidate
            candidate_idx = candidates[i_]
            candidate = x[order[candidate_idx]]

            # Update the accumulators and compute delta_b
            diff = last_candidate - candidate

            # What follows is a section of code that has been optimized for
            # speed at the expense of some readability.  To make it easier to
            # understand this code in the future, I have included a
            # "simple" block that implements the same math in a more
            # straightforward (but much less efficient) way.
            # The (commented out) code between "BEGIN SIMPLE" and "END SIMPLE"
            # should produce the same output as the code between
            # "BEGIN HYPER-OPTIMIZED" and "END HYPER-OPTIMIZED".

            # BEGIN SIMPLE
            # Calculate delta_b
            #            for j  in range(0,last_candidate_idx+1):
            #                delta_b[j] = diff
            #            for j in range(last_candidate_idx+1,candidate_idx):
            #                float_tmp = (X[j,variable] - candidate) * b_parent[j]
            #                delta_b[j] = float_tmp
            #
            # Update u and z_end_squared
            #            u[0:k+1] += np.dot(delta_b,B_orth[:,0:k+1])
            #            u_end += 2*np.dot(delta_b,b) + np.dot(delta_b, delta_b)
            #
            # Update c_end
            #            c_end += np.dot(delta_b,y)
            #
            # Update z_end_squared
            #            z_end_squared = ((c_end - np.dot(u[0:k+1],c[0:k+1]))**2) / (u_end)
            #
            # Update b
            #            b += delta_b
            # END SIMPLE

            # BEGIN HYPER-OPTIMIZED
            delta_b_squared = 0.0


            for p in range(self.y.shape[1]):
                delta_c_end[p] = 0.0
            delta_u_end = 0.0

            # Update the accumulators
            for j_ in range(last_last_candidate_idx + 1,
                            last_candidate_idx + 1):
                j = order[j_]
                y_cum += y[j, 0]
                for h in range(k + 1):  # TODO: BLAS
                    B_orth_times_parent_cum[h] += B_orth[j, h] * b_parent[j]
                b_times_parent_cum += b[j] * b_parent[j]
                parent_squared_cum += b_parent[j] ** 2
                for p in range(self.y.shape[1]):
                    parent_times_y_cum[p] +=  b_parent[j] * y[j, p]
            for p in range(self.y.shape[1]):
                delta_c_end[p] += diff * parent_times_y_cum[p]
            delta_u_end += 2 * diff * b_times_parent_cum
            delta_b_squared = (diff ** 2) * parent_squared_cum

            # Update u and a bunch of other stuff
            for j in range(k + 1):
                float_tmp = diff * B_orth_times_parent_cum[j]

                for p in range(self.y.shape[1]):
                    u_dot_c[p] += float_tmp * c[j, p] 
                u_dot_u += 2 * u[j] * float_tmp + float_tmp * float_tmp
                u[j] += float_tmp
            for j_ in range(last_candidate_idx + 1, candidate_idx):
                j = order[j_]
                delta_b_j = (x[j] - candidate) * b_parent[j]
                delta_b_squared += delta_b_j ** 2

                for p in range(self.y.shape[1]):
                    delta_c_end[p] += delta_b_j * y[j, p]
                delta_u_end += 2 * delta_b_j * b[j]
                for h in range(k + 1):
                    float_tmp = delta_b_j * B_orth[j, h]

                    for p in range(self.y.shape[1]):
                        u_dot_c[p] +=  float_tmp * c[h, p]
                    u_dot_u += 2 * u[h] * float_tmp + float_tmp * float_tmp
                    u[h] += float_tmp
                b[j] += delta_b_j

            # Update u_end
            delta_u_end += delta_b_squared
            u_end += delta_u_end

            # Update c_end

            for p in range(self.y.shape[1]):
                c_end[p] += delta_c_end[p]

            # Update b_times_parent_cum
            b_times_parent_cum += parent_squared_cum * diff

            # Compute the new z_end_squared (this is the quantity we're
            # optimizing)
            if (u_end - u_dot_u) <= self.zero_tol:
                continue
            z_end_squared = 0.

            for p in range(self.y.shape[1]):
                z_end_squared += ((c_end[p] - u_dot_c[p]) ** 2) * output_weight[p]
            z_end_squared /= (u_end - u_dot_u)
            # END HYPER-OPTIMIZED

            # Update the best if necessary
            if z_end_squared > best_z_end_squared:
                best_z_end_squared = z_end_squared
                best_candidate_idx = candidate_idx
                best_candidate = candidate

        # Compute the mse for the best z_end and set return values
        mse[0] = (
            self.y_squared - self.c_squared - best_z_end_squared) / self.m
        knot[0] = best_candidate
        knot_idx[0] = best_candidate_idx
