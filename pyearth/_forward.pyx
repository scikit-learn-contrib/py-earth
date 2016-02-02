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
from ._knot_search cimport knot_search, MultipleOutcomeDependentData, PredictorDependentData, \
    KnotSearchReadOnlyData, KnotSearchState, KnotSearchWorkingData, KnotSearchData
import sys
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
                 cnp.ndarray[FLOAT_t, ndim=2] sample_weight,
                 **kwargs):
        
        cdef INDEX_t i
        self.X = X
        self.missing = missing
        self.y = y
        # Assuming Earth.fit got capital W (the inverse of squared variance)
        # so the objective function is (sqrt(W) * residual) ^ 2)
        self.sample_weight = np.sqrt(sample_weight)
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan       = kwargs.get('endspan', -1)
        self.minspan       = kwargs.get('minspan', -1)
        self.endspan_alpha = kwargs.get('endspan_alpha', .05)
        self.minspan_alpha = kwargs.get('minspan_alpha', .05)
        self.max_terms     = kwargs.get('max_terms', 2 * self.n + self.m // 10)
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
            
        self.last_fast_empty = False
        self.last_fast_low_improvement = False
        self.fast_heap = []

        if self.xlabels is None:
            self.xlabels = ['x' + str(i) for i in range(self.n)]
        if self.check_every < 0:
            self.check_every = (<int > (self.m / self.min_search_points)
                                if self.m > self.min_search_points
                                else 1)
        
        weighted_mean = np.mean((self.sample_weight ** 2) * self.y)
        self.sst = np.sum((self.sample_weight * (self.y - weighted_mean)) ** 2)

        self.record = ForwardPassRecord(
            self.m, self.n, self.penalty, self.sst / np.sum(self.sample_weight ** 2), self.xlabels)
        self.basis = Basis(self.n)
        self.basis.append(ConstantBasisFunction())
        if self.use_fast is True:
            content = FastHeapContent(idx=0)
            heappush(self.fast_heap, content)
            print 'push', content
            
        self.mwork = np.empty(shape=self.m, dtype=np.int)
        
        self.B = np.ones(
            shape=(self.m, self.max_terms + 4), order='F', dtype=np.float)
        self.basis.transform(self.X, self.missing, self.B[:,0:1])
        
        if self.endspan < 0:
            self.endspan = round(3 - log2(self.endspan_alpha / self.n))
        
        self.linear_variables = np.zeros(shape=self.n, dtype=np.int)
        self.init_linear_variables()
        
        
        
        # Removed in favor of new knot search code
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
        
        # Initialize the data structures for knot search
        self.n_outcomes = self.y.shape[1]
        n_predictors = self.X.shape[1]
        n_weights = self.sample_weight.shape[1]
        self.workings = []
        self.outcome = MultipleOutcomeDependentData.alloc(self.y, self.sample_weight, self.m, 
                                                          self.n_outcomes, self.max_terms + 4)
        self.outcome.update_from_array(self.B[:,0], self.zero_tol)
        for i in range(self.n_outcomes):
            working = KnotSearchWorkingData.alloc(self.max_terms + 4)
            self.workings.append(working)
        self.predictors = []
        for i in range(n_predictors):
            x = self.X[:, i]
            x[missing[:,i]==1] = 0.
            predictor = PredictorDependentData.alloc(x)
            self.predictors.append(predictor)

    cpdef Basis get_basis(ForwardPasser self):
        return self.basis

    cpdef init_linear_variables(ForwardPasser self):
        cdef INDEX_t variable
        cdef cnp.ndarray[INT_t, ndim = 1] order
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef ConstantBasisFunction root_basis_function = self.basis[0]
        for variable in range(self.n):
            order = np.argsort(X[:, variable])[::-1].astype(np.int)
            if root_basis_function.valid_knots(B[order, 0], X[order, variable],
                                               variable, self.check_every,
                                               self.endspan, self.minspan,
                                               self.minspan_alpha, self.n,
                                               self.mwork).shape[0] == 0:
                linear_variables[variable] = 1
            else:
                linear_variables[variable] = 0
                
    cpdef run(ForwardPasser self):
        if self.max_terms > 1:
            while True:
                self.next_pair()
                if self.stop_check():
                    break
                self.iteration_number += 1

    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1
        if self.record.iterations[last].get_size() > self.max_terms:
            self.record.stopping_condition = MAXTERMS
            return True
        rsq = self.record.rsq(last)
        if rsq > 1 - self.thresh:
            self.record.stopping_condition = MAXRSQ
            return True
        previous_rsq = self.record.rsq(last - 1)
        if rsq - previous_rsq < self.thresh:
            if self.use_fast and not self.last_fast_low_improvement:
                self.last_fast_low_improvement = True
            else:
                self.record.stopping_condition = NOIMPRV
                return True
        elif self.use_fast:
            self.last_fast_low_improvement = False
        if self.record.grsq(last) < -10:
            self.record.stopping_condition = LOWGRSQ
            return True
        if self.record.iterations[last].no_further_candidates():
            self.record.stopping_condition = NOCAND
            return True
        return False
    
    cpdef orthonormal_update(ForwardPasser self, b):
        # Update the outcome data
        linear_dependence = False
        return_codes = []
        return_code = self.outcome.update_from_array(b, self.zero_tol)
        if return_code == -1:
            raise ValueError('This should not have happened.')
        if return_code == 1:
            linear_dependence = True
        return linear_dependence
#         for outcome in self.outcomes:
#             return_code = outcome.update_from_array(b, self.zero_tol)
#             if return_code == -1:
#                 raise ValueError('This should not have happened.')
#             return_codes.append(return_code != 0)
#         # TODO: Change to any?
#         if all(return_codes):
#             linear_dependence = True
#         return linear_dependence
    
    cpdef orthonormal_downdate(ForwardPasser self):
        self.outcome.downdate()
        
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
        cdef int knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef FLOAT_t mse_choice_cur_parent
        cdef int variable_choice_cur_parent
        cdef int knot_idx_choice
        cdef INDEX_t parent_idx_choice
        cdef BasisFunction parent_choice
        cdef BasisFunction new_parent
        cdef BasisFunction new_basis_function
        parent_basis_content_choice = None
        parent_basis_content = None
        cdef INDEX_t variable_choice
        cdef bint first = True
#         cdef BasisFunction bf1
#         cdef BasisFunction bf2
#         cdef BasisFunction bf3
#         cdef BasisFunction bf4
        cdef bint already_covered
        cdef INDEX_t k = len(self.basis)
        cdef INDEX_t endspan
        cdef bint linear_dependence
        cdef bint dependent
        # TODO: Shouldn't there be weights here?
        cdef FLOAT_t gcv_factor_k_plus_1 = gcv_adjust(k + 1, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_2 = gcv_adjust(k + 2, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_3 = gcv_adjust(k + 3, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_4 = gcv_adjust(k + 4, self.m,
                                                      self.penalty)
        cdef FLOAT_t gcv_
        cdef FLOAT_t mse_
        cdef INDEX_t i
        cdef bint eligible
        cdef bint covered
        cdef bint missing_flag
        cdef bint choice_needs_coverage
#         cdef int max_variable_degree
        
        cdef cnp.ndarray[FLOAT_t, ndim = 2] X = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.X)
        cdef cnp.ndarray[BOOL_t, ndim = 2] missing = (
            <cnp.ndarray[BOOL_t, ndim = 2] > self.missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 2] B = (
            <cnp.ndarray[FLOAT_t, ndim = 2] > self.B)
        cdef cnp.ndarray[INT_t, ndim = 1] linear_variables = (
            <cnp.ndarray[INT_t, ndim = 1] > self.linear_variables)
        cdef cnp.ndarray[BOOL_t, ndim = 1] has_missing = (
            <cnp.ndarray[BOOL_t, ndim = 1] > self.has_missing)
        cdef cnp.ndarray[FLOAT_t, ndim = 1] b
        cdef cnp.ndarray[FLOAT_t, ndim = 1] p
        
        if self.use_fast and not (self.last_fast_empty or self.last_fast_low_improvement):
            # choose only among the top "fast_K" basis functions
            # as parents
            nb_basis = min(self.fast_K, k, len(self.fast_heap))
        else:
            nb_basis = min(k, len(self.fast_heap))

        content_to_be_repushed = []
        for idx in range(nb_basis):
            # Iterate over parents
            if self.use_fast:
                # retrieve the next basis function to try as parent
                parent_basis_content = heappop(self.fast_heap)
                print 'pop', parent_basis_content
                content_to_be_repushed.append(parent_basis_content)
                parent_idx = parent_basis_content.idx
                mse_choice_cur_parent = -1
                variable_choice_cur_parent = -1
            else:
                parent_idx = idx

            parent = self.basis.get(parent_idx)
            if not parent.is_splittable():
                continue

            if self.use_fast:
                # each "fast_h" iteration, force to pass through all the variables,
                if self.iteration_number - parent_basis_content.m > self.fast_h or self.last_fast_empty:
                    variables = range(self.n)
                    parent_basis_content.m = self.iteration_number
                # in the opposite case, just use the last chosen variable
                else:
                    variables = [parent_basis_content.v]
                variables = range(self.n)
            else:
                variables = range(self.n)
            
            parent_degree = parent.effective_degree()
            
            for variable in variables:
                # Determine whether missingness needs to be accounted for.
                if self.allow_missing and has_missing[variable]:
                    missing_flag = True
                    eligible = parent.eligible(variable)
                    covered = parent.covered(variable)
                else:
                    missing_flag = False
                
                # Make sure not to exceed max_degree (but don't count the 
                # covering missingness basis function if required)
                if self.max_degree >= 0:
                    if parent_degree >= self.max_degree:
                        continue
                
                # If there is missing data and this parent is not 
                # an eligible parent for this variable with missingness
                # (because it includes a non-missing factor for the variable)
                # then skip this variable.
                if missing_flag and not eligible:
                    continue

                # Add the linear term to B
                predictor = self.predictors[variable]
                
#                 # If necessary, protect from missing data
#                 if missing_flag:
#                     B[missing[:, variable]==1, k] = 0.
#                     b = B[:, k]
#                     # Update the outcome data
#                     linear_dependence = self.orthonormal_update(b)
                    
                if missing_flag and not covered:
                    p = B[:, parent_idx] * (1 - missing[:, variable])
                    b = B[:, parent_idx] * (1 - missing[:, variable])
                    self.orthonormal_update(b)
                    b = B[:, parent_idx] * missing[:, variable]
                    self.orthonormal_update(b)
                    q = k + 3
                else:
                    p = self.B[:, parent_idx]
                    q = k + 1
                
                b = p * predictor.x
                if missing_flag and not covered:
                    b[missing[:, variable] == 1] = 0
                linear_dependence = self.orthonormal_update(b)

                # If a new hinge function does not improve the gcv over the
                # linear term then just the linear term will be retained
                # (if allow_linear).  Calculate the gcv with just the linear
                # term in order to compare later.  Note that the mse with
                # another term never increases, but the gcv may because it
                # penalizes additional terms.
                mse_ = sum(self.outcome.sse()) / np.sum(self.sample_weight ** 2)
                if missing_flag and not covered:
                    gcv_ = gcv_factor_k_plus_3 * mse_
                else:
                    gcv_ = gcv_factor_k_plus_1 * mse_

                if linear_variables[variable]:
                    mse = mse_
                    knot_idx = -1
                else:
                    # Find the valid knot candidates
                    candidates, candidates_idx = predictor.knot_candidates(p, self.endspan,
                                                                           self.minspan, 
                                                                           self.minspan_alpha,
                                                                           self.n, set(parent.knots(variable)))
                    # Choose the best candidate (if no candidate is an
                    # improvement on the linear term in terms of gcv, knot_idx
                    # is set to -1
                    if len(candidates_idx) > 0:
#                         candidates = np.array(predictor.x)[candidates_idx]
                    
                        # Find the best knot location for this parent and
                        # variable combination
                        # Assemble the knot search data structure
                        constant = KnotSearchReadOnlyData(predictor, self.outcome)
                        search_data = KnotSearchData(constant, self.workings, q)

                        # Run knot search
#                         print len(candidates_idx)
                        knot, knot_idx, mse = knot_search(search_data, candidates, p, q, 
                                                          self.m, len(candidates), self.n_outcomes)
                        mse /= np.sum(self.sample_weight ** 2)
#                         print knot_idx
                        knot_idx = candidates_idx[knot_idx]
#                         print parent, variable, mse
#                         print knot_idx
#                         mse = mse ** 2
                        
                        # If the hinge function does not decrease the gcv then
                        # just keep the linear term (if allow_linear is True)
                        if self.allow_linear:
                            if missing_flag and not covered:
                                if gcv_factor_k_plus_4 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                            else:
                                if gcv_factor_k_plus_2 * mse >= gcv_:
                                    mse = mse_
                                    knot_idx = -1
                    else:
                        # Do an orthonormal downdate and skip to the next
                        # iteration
                        if missing_flag and not covered:
                            self.orthonormal_downdate()
                            self.orthonormal_downdate()
                        self.orthonormal_downdate()
                        continue
                        # TODO: Should that continue be there?
                        
                # Do an orthonormal downdate
                if missing_flag and not covered:
                    self.orthonormal_downdate()
                    self.orthonormal_downdate()
                self.orthonormal_downdate()
                
                # Update the choices
#                 print parent, variable, mse, mse_choice, knot_idx, knot_idx_choice
                if mse < mse_choice or first:
#                     print 'choose'
                    if first:
                        first = False
                        self.last_fast_empty = False
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
#                     print 'knot_idx_choice', knot_idx_choice
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
                    if (mse_choice_cur_parent == -1) or \
                       (mse < mse_choice_cur_parent):
                        mse_choice_cur_parent = mse
                        variable_choice_cur_parent = variable
            if self.use_fast is True:
                if mse_choice_cur_parent != -1:
                    parent_basis_content.mse = mse_choice_cur_parent
                    parent_basis_content.v = variable_choice_cur_parent
        
        if self.use_fast is True:
            for content in content_to_be_repushed:
                print 'push', content
                heappush(self.fast_heap, content)

        # Make sure at least one candidate was checked
        if first:
            if self.use_fast and not self.last_fast_empty:
                self.last_fast_empty = True
                return
            else:
                self.record[len(self.record) - 1].set_no_candidates(True)
                return
        
        # Add the new basis functions
        label = self.xlabels[variable_choice]
        print 'Chose variable %s and parent %s' % (label, parent_choice)
        if self.use_fast is True: 
            parent_basis_content_choice.m = -np.inf
        if choice_needs_coverage:
            new_parent = parent_choice.get_coverage(variable_choice)
            if new_parent is None:
                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               True, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                    print 'push', content
                self.basis.append(new_basis_function)
                new_parent = new_basis_function
                
                new_basis_function = MissingnessBasisFunction(parent_choice, variable_choice,
                                               False, label)
                new_basis_function.apply(X, missing, B[:, len(self.basis)])
                self.orthonormal_update(B[:, len(self.basis)])
                if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                    content = FastHeapContent(idx=len(self.basis))
                    heappush(self.fast_heap, content)
                    print 'push', content
                self.basis.append(new_basis_function)
#             if self.basis.has_coverage(variable_choice):
#                 bf3, bf4 = self.basis.get_coverage(variable_choice)
#                 already_covered = True
#             else:
#                 already_covered = False
#             parent_choice = bf3
        else:
            new_parent = parent_choice
        if knot_idx_choice != -1:
            # Add the new basis functions
            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     False, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, FastHeapContent(idx=len(self.basis)))
                print 'push', content
            self.basis.append(new_basis_function)
            
            new_basis_function = HingeBasisFunction(new_parent,
                                     knot_choice, knot_idx_choice,
                                     variable_choice,
                                     True, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
                print 'push', content
            self.basis.append(new_basis_function)
            
#             bf1.apply(X, missing, B[:, k])
#             bf2.apply(X, missing, B[:, k + 1])
# 
#             self.basis.append(bf1)
#             print 'append %s' % str(bf1)
#             self.basis.append(bf2)
#             print 'append %s' % str(bf2)
#             
#             if choice_needs_coverage:
#                 print 'choice needs coverage'
#                 if not already_covered:
#                     bf3.apply(X, missing, B[:, k + 2])
#                     bf4.apply(X, missing, B[:, k + 3])
#                     self.basis.add_coverage(variable_choice, bf3, bf4)
#                     print 'append %s' % str(bf3)
#                     print 'append %s' % str(bf4)

#             if self.use_fast is True:
#                 bf1_content = FastHeapContent(idx=k)
#                 heappush(self.fast_heap, bf1_content)
#                 bf2_content = FastHeapContent(idx=k + 1)
#                 heappush(self.fast_heap, bf2_content)
#                 if choice_needs_coverage:
#                     if not already_covered:
#                         bf3_content = FastHeapContent(idx=k + 2)
#                         heappush(self.fast_heap, FastHeapContent(idx=k + 2))
#                         bf4_content = FastHeapContent(idx=k + 3)
#                         heappush(self.fast_heap, FastHeapContent(idx=k + 3))
                    
#             # Orthogonalize the new basis
#             if self.orthonormal_update(B[:, k]) == 1:
#                 bf1.make_unsplittable()
#             if self.orthonormal_update(B[:, k + 1]) == 1:
#                 bf2.make_unsplittable()
#             if choice_needs_coverage:
#                 if not already_covered:
#                     if self.orthonormal_update(B[:, k + 2]) == 1:
#                         pass
#                     if self.orthonormal_update(B[:, k + 3]) == 1:
#                         pass
        elif not dependent and knot_idx_choice == -1:
            # In this case, only add the linear basis function (in addition to 
            # covering missingness basis functions if needed)
#             if choice_needs_coverage:
#                 bf2 = MissingnessBasisFunction(parent_choice, variable_choice,
#                                                True, label)
#                 bf3 = MissingnessBasisFunction(parent_choice, variable_choice,
#                                                False, label)
#                 if self.basis.has_coverage(variable_choice):
#                     bf2, bf3 = self.basis.get_coverage(variable_choice)
#                     already_covered = True
#                 else:
#                     already_covered = False
#                 parent_choice = bf2
            new_basis_function = LinearBasisFunction(parent_choice, variable_choice, label)
            new_basis_function.apply(X, missing, B[:, len(self.basis)])
            self.orthonormal_update(B[:, len(self.basis)])
            if self.use_fast and new_basis_function.is_splittable() and new_basis_function.effective_degree() < self.max_degree:
                content = FastHeapContent(idx=len(self.basis))
                heappush(self.fast_heap, content)
                print 'push', content
            self.basis.append(new_basis_function)
            
#             bf1.apply(X, missing, B[:, k])
#             self.basis.append(bf1)
#             print 'append %s' % str(bf1)
#             if choice_needs_coverage:
#                 if not already_covered:
#                     bf2.apply(X, missing, B[:, k + 1])
#                     bf3.apply(X, missing, B[:, k + 2])
#                     self.basis.add_coverage(variable_choice, bf2, bf3)
#                     print 'append %s' % str(bf2)
#                     print 'append %s' % str(bf3)
#             if self.use_fast is True:
#                 bf1_content = FastHeapContent(idx=k)
#                 heappush(self.fast_heap, bf1_content)
#                 if choice_needs_coverage:
#                     if not already_covered:
#                         bf2_content = FastHeapContent(idx=k + 1)
#                         heappush(self.fast_heap, bf2_content)
#                         bf3_content = FastHeapContent(idx=k + 2)
#                         heappush(self.fast_heap, bf3_content)
#             # Orthogonalize the new basis
#             if self.orthonormal_update(B[:, k]) == 1:
#                 bf1.make_unsplittable()
#             if choice_needs_coverage:
#                 if not already_covered:
#                     if self.orthonormal_update(B[:, k + 1]) == 1:
#                         pass
#                     if self.orthonormal_update(B[:, k + 2]) == 1:
#                         pass
        else:  # dependent and knot_idx_choice == -1
            # In this case there were no acceptable choices remaining, so end
            # the forward pass
            self.record[len(self.record) - 1].set_no_candidates(True)
            return
        

        # Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,
                                                variable_choice,
                                                knot_idx_choice, mse_choice,
                                                len(self.basis)))
