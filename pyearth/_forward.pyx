# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

from _util cimport reorderxby, gcv_adjust
from _basis cimport Basis, BasisFunction, ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
from _record cimport ForwardPassIteration

from libc.math cimport sqrt, abs, log, log2
import numpy as np
import scipy.linalg
cnp.import_array()
cdef class ForwardPasser:
    
    def __init__(ForwardPasser self, cnp.ndarray[FLOAT_t, ndim=2] X, cnp.ndarray[FLOAT_t, ndim=1] y, **kwargs):
        cdef unsigned int i
        cdef FLOAT_t sst
        self.X = X
        self.y = y
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan = kwargs['endspan'] if 'endspan' in kwargs else -1
        self.minspan = kwargs['minspan'] if 'minspan' in kwargs else -1
        self.endspan_alpha = kwargs['endspan_alpha'] if 'endspan_alpha' in kwargs else .05
        self.minspan_alpha = kwargs['minspan_alpha'] if 'minspan_alpha' in kwargs else .05
        self.max_terms = kwargs['max_terms'] if 'max_terms' in kwargs else 10
        self.max_degree = kwargs['max_degree'] if 'max_degree' in kwargs else 1
        self.thresh = kwargs['thresh'] if 'thresh' in kwargs else 0.001
        self.penalty = kwargs['penalty'] if 'penalty' in kwargs else 3.0
        self.check_every = kwargs['check_every'] if 'check_every' in kwargs else -1
        self.min_search_points = kwargs['min_search_points'] if 'min_search_points' in kwargs else 100
        self.xlabels = kwargs['xlabels'] if 'xlabels' in kwargs else ['x'+str(i) for i in range(self.n)]
        if self.check_every < 0:
            self.check_every = <int> (self.m / self.min_search_points) if self.m > self.min_search_points else 1
        self.sst = np.sum((self.y - np.mean(y))**2)/self.m
        self.y_squared = np.dot(self.y,self.y)
        self.record = ForwardPassRecord(self.m,self.n,self.penalty,self.sst)
        self.basis = Basis()
        self.basis.append(ConstantBasisFunction())
        
        self.sorting = np.empty(shape=self.m, dtype=np.int)
        self.mwork = np.empty(shape=self.m, dtype=np.int)
        self.u = np.empty(shape=self.max_terms, dtype=float)
        self.B_orth_times_parent_cum = np.empty(shape=self.max_terms,dtype=np.float)
        self.B = np.ones(shape=(self.m,self.max_terms), order='C',dtype=np.float)
        self.B_orth = np.ones(shape=(self.m,self.max_terms), order='C',dtype=np.float)
        self.u = np.empty(shape=self.max_terms, dtype=np.float)
        self.c = np.empty(shape=self.max_terms, dtype=np.float)
        self.norms = np.empty(shape=self.max_terms, dtype=np.float)
        self.c_squared = 0.0
        self.sort_tracker = np.empty(shape=self.m, dtype=np.int)
        for i in range(self.m):
            self.sort_tracker[i] = i
        self.zero_tol = 1e-6
        
#        self.x_order = np.empty(shape=(self.m,self.n),dtype=np.int)
        self.linear_variables = np.zeros(shape=self.n,dtype=np.int)
#        self.init_x_order()
        self.init_linear_variables()
        
        #Add in user selected linear variables
        if 'linvars' in kwargs:
            for linvar in kwargs['linvars']:
                if linvar in self.xlabels:
                    self.linear_variables[self.xlabels.index(linvar)] = 1
                elif linvar in range(self.n):
                    self.linear_variables[linvar] = 1
                else:
                    raise IndexError('Unknown variable selected in linvars argument.')
        
        #Initialize B_orth, c, and c_squared (assuming column 0 of B_orth is already filled with 1)
        self.orthonormal_update(0)
    
    cpdef Basis get_basis(ForwardPasser self):
        return self.basis
    
#    cpdef init_x_order(ForwardPasser self):
#        cdef unsigned int variable
#        cdef cnp.ndarray[FLOAT_t, ndim=2] X = <cnp.ndarray[FLOAT_t, ndim=2]> self.X
#        cdef cnp.ndarray[INT_t, ndim=2] x_order = <cnp.ndarray[INT_t, ndim=2]> self.x_order
#        for variable in range(self.n):
#            x_order[:,variable] = np.argsort(X[:,variable])[::-1]
    
    cpdef init_linear_variables(ForwardPasser self):
        cdef unsigned int variable
        cdef unsigned int endspan
#        cdef cnp.ndarray[INT_t, ndim=2] x_order = <cnp.ndarray[INT_t, ndim=2]> self.x_order
        cdef cnp.ndarray[INT_t, ndim=1] order
        cdef cnp.ndarray[INT_t, ndim=1] linear_variables = <cnp.ndarray[INT_t, ndim=1]> self.linear_variables
        cdef cnp.ndarray[FLOAT_t, ndim=2] B = <cnp.ndarray[FLOAT_t, ndim=2]> self.B
        cdef cnp.ndarray[FLOAT_t, ndim=2] X = <cnp.ndarray[FLOAT_t, ndim=2]> self.X
        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha/self.n))
        cdef ConstantBasisFunction root_basis_function = self.basis[0]
        for variable in range(self.n):
            order = np.argsort(X[:,variable])[::-1]
            if root_basis_function.valid_knots(B[order,0], X[order,variable], 
                                               variable, self.check_every, endspan, 
                                               self.minspan, self.minspan_alpha, 
                                               self.n, self.mwork).shape[0] == 0:
                linear_variables[variable] = 1
            else:
                linear_variables[variable] = 0
    
    def get_B_orth(ForwardPasser self):
        return self.B_orth
    
    cpdef run(ForwardPasser self):
        cdef unsigned int i
        while True:
            self.next_pair()
            if self.stop_check():
                break

    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1
        if self.record.iterations[last].get_size() + 2 > self.max_terms:
            self.record.stopping_condition = MAXTERMS
            return True
        rsq = self.record.rsq(last)
        if rsq > 1 - self.thresh:
            self.record.stopping_condition = MAXRSQ
            return True
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
    
    cpdef int orthonormal_update(ForwardPasser self, unsigned int k):
        '''Orthogonalize and normalize column k of B_orth against all previous columns of B_orth.'''
        #Currently implemented using modified Gram-Schmidt process
        #TODO: Optimize - replace calls to numpy with calls to blas
        
        cdef cnp.ndarray[FLOAT_t, ndim=2] B_orth = <cnp.ndarray[FLOAT_t, ndim=2]> self.B_orth
        cdef cnp.ndarray[FLOAT_t, ndim=1] c = <cnp.ndarray[FLOAT_t, ndim=1]> self.c
        cdef cnp.ndarray[FLOAT_t, ndim=1] y = <cnp.ndarray[FLOAT_t, ndim=1]> self.y
        cdef cnp.ndarray[FLOAT_t, ndim=1] norms = <cnp.ndarray[FLOAT_t, ndim=1]> self.norms
        
        cdef unsigned int i
        cdef FLOAT_t nrm
        cdef FLOAT_t nrm0
        
        #Get the original norm
        nrm0 = sqrt(np.dot(B_orth[:,k],B_orth[:,k]))
        
        #Orthogonalize
        if k > 0:
            for i in range(k):
                B_orth[:,k] -= B_orth[:,i] * (np.dot(B_orth[:,k],B_orth[:,i]))
        
        #Normalize
        nrm = sqrt(np.dot(B_orth[:,k],B_orth[:,k]))
        norms[k] = nrm
        if nrm0 <= self.zero_tol or nrm/nrm0 <= self.zero_tol:
            B_orth[:,k] = 0
            c[k] = 0
            return 1 #The new column is in the column space of the previous columns
        B_orth[:,k] /= nrm
        
        #Update c
        c[k] = np.dot(B_orth[:,k],y)
        self.c_squared += c[k]**2
        
        return 0 #No problems
    
    cpdef orthonormal_downdate(ForwardPasser self, unsigned int k):
        '''
        Undo the effects of the last orthonormal update.  You can only undo the last orthonormal update this way.
        There will be no warning of any kind if you mess this up.  You'll just get wrong answers.
        In reality, all this does is downdate c_squared (the elements of c and B_orth are left alone, since they
        can simply be ignored until they are overwritten).
        '''
        self.c_squared -= self.c[k]**2
        
    def trace(self):
        return self.record
        
    cdef next_pair(ForwardPasser self):
        cdef unsigned int variable
        cdef unsigned int parent_idx
        cdef unsigned int parent_degree
        cdef unsigned int nonzero_count
        cdef BasisFunction parent
        cdef cnp.ndarray[INT_t,ndim=1] candidates_idx
        cdef FLOAT_t knot
        cdef FLOAT_t mse
        cdef unsigned int knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef int knot_idx_choice
        cdef unsigned int parent_idx_choice
        cdef BasisFunction parent_choice
        cdef unsigned int variable_choice
        cdef bint first = True
        cdef BasisFunction bf1
        cdef BasisFunction bf2
        cdef unsigned int k = len(self.basis)
        cdef unsigned int endspan
        cdef bint linear_dependence
        cdef bint dependent
        cdef FLOAT_t gcv_factor_k_plus_1 = gcv_adjust(k+1,self.m,self.penalty)
        cdef FLOAT_t gcv_factor_k_plus_2 = gcv_adjust(k+2,self.m,self.penalty)
        cdef FLOAT_t gcv_
        cdef FLOAT_t mse_
        
        cdef cnp.ndarray[FLOAT_t,ndim=2] X = <cnp.ndarray[FLOAT_t,ndim=2]> self.X
        cdef cnp.ndarray[FLOAT_t,ndim=2] B = <cnp.ndarray[FLOAT_t,ndim=2]> self.B
        cdef cnp.ndarray[FLOAT_t,ndim=2] B_orth = <cnp.ndarray[FLOAT_t,ndim=2]> self.B_orth
        cdef cnp.ndarray[FLOAT_t,ndim=1] y = <cnp.ndarray[FLOAT_t,ndim=1]> self.y
        cdef cnp.ndarray[INT_t,ndim=1] linear_variables = <cnp.ndarray[INT_t,ndim=1]> self.linear_variables
        
        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha/self.n))
        
        #Iterate over variables
        for variable in range(self.n):
            
            #Sort the data
            self.sorting[:] = np.argsort(X[:,variable])[::-1] #TODO: eliminate Python call / data copy
            reorderxby(X,B,B_orth,y,self.sorting,self.sort_tracker)
            
            #Iterate over parents
            for parent_idx in range(k):
                linear_dependence = False
                
                parent = self.basis.get(parent_idx)
                if self.max_degree >= 0:
                    parent_degree = parent.degree()
                    if parent_degree >= self.max_degree:
                        continue
                if not parent.is_splittable():
                    continue
                
                #Add the linear term to B
                B[:,k] = B[:,parent_idx]*X[:,variable] #TODO: Optimize
                
                #Find the valid knot candidates
                candidates_idx = parent.valid_knots(B[:,parent_idx], X[:,variable], variable, self.check_every, endspan, self.minspan, self.minspan_alpha, self.n, self.mwork)

                #Orthonormalize
                B_orth[:,k] = B[:,k]
                linear_dependence = self.orthonormal_update(k)
                
                #If a new hinge function does not improve the gcv over the linear term
                #then just the linear term will be retained.  Calculate the gcv with 
                #just the linear term in order to compare later.  Note that the mse with 
                #another term never increases, but the gcv may because it penalizes additional
                #terms.
                mse_ = (self.y_squared - self.c_squared) / self.m
                gcv_ = gcv_factor_k_plus_1*(self.y_squared - self.c_squared) / self.m
                
                if linear_variables[variable]:
                    mse = mse_
                    knot_idx = -1
                
                elif len(candidates_idx) > 0:
                #Choose the best candidate (if no candidate is an improvement on the linear term in terms of gcv, knot_idx is set to -1

                    #Find the best knot location for this parent and variable combination
                    self.best_knot(parent_idx,variable,k,candidates_idx,&mse,&knot,&knot_idx)
                    
                    #If the hinge function does not decrease the gcv then just keep the linear term
                    if gcv_factor_k_plus_2*mse >= gcv_:
                        mse = mse_
                        knot_idx = -1
                    
                else:
                    #Do an orthonormal downdate
                    self.orthonormal_downdate(k)
                    continue
                
                #Do an orthonormal downdate
                self.orthonormal_downdate(k)
                
                #Update the choices
                if first:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    variable_choice = variable
                    first = False
                    dependent = linear_dependence
                if mse < mse_choice:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    variable_choice = variable
                    dependent = linear_dependence
                    
        #Make sure at least one candidate was checked
        if first:
            self.record[-1].set_no_candidates(True)
            return
        
        #Add the new basis functions
        parent = self.basis.get(parent_idx)
        label = self.xlabels[variable_choice]
        if knot_idx_choice != -1:
            bf1 = HingeBasisFunction(parent_choice,knot_choice,knot_idx_choice,variable_choice,False,label)
            bf2 = HingeBasisFunction(parent_choice,knot_choice,knot_idx_choice,variable_choice,True,label)
            bf1.apply(X,B[:,k])
            bf2.apply(X,B[:,k+1])
            self.basis.append(bf1)
            self.basis.append(bf2)
            #Orthogonalize the new basis
            B_orth[:,k] = B[:,k]
            if self.orthonormal_update(k) == 1:
                bf1.make_unsplittable()
            B_orth[:,k+1] = B[:,k+1]
            if self.orthonormal_update(k+1) == 1:
                bf2.make_unsplittable()
#        elif knot_idx_choice != -1:
#            bf1 = HingeBasisFunction(parent_choice,knot_choice,knot_idx_choice,variable_choice,False,label)
#            bf1.apply(X,B[:,k])
#            self.basis.append(bf1)
#            #Orthogonalize the new basis
#            B_orth[:,k] = B[:,k]
#            if self.orthonormal_update(k) == 1:
#                pass
##                bf1.make_unsplittable()
        elif not dependent and knot_idx_choice == -1:
            bf1 = LinearBasisFunction(parent_choice,variable_choice,label)
            bf1.apply(X,B[:,k])
            self.basis.append(bf1)
            #Orthogonalize the new basis
            B_orth[:,k] = B[:,k]
            if self.orthonormal_update(k) == 1:
                bf1.make_unsplittable()
        else:#dependent and knot_idx_choice == -1
            self.record[-1].set_no_candidates(True)
            return
            
        #TODO: Undo the sorting
        
        #Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,variable_choice,knot_idx_choice,mse_choice,len(self.basis)))
        
    cdef best_knot(ForwardPasser self, unsigned int parent, unsigned int variable, unsigned int k, cnp.ndarray[INT_t,ndim=1] candidates, FLOAT_t * mse, FLOAT_t * knot, unsigned int * knot_idx):
        '''
        Find the best knot location (in terms of squared error).
        
        Assumes:
        B[:,k] is the linear term for variable
        X[:,variable] is in decreasing order
        candidates is in increasing order (it is an array of indices into X[:,variable]
        mse is a pointer to the mean squared error of including just the linear term in B[:,k]
        '''
        
        cdef cnp.ndarray[FLOAT_t, ndim=1] b = <cnp.ndarray[FLOAT_t, ndim=1]> self.B[:,k+1]
        cdef cnp.ndarray[FLOAT_t, ndim=1] b_parent = <cnp.ndarray[FLOAT_t, ndim=1]> self.B[:,parent]
        cdef cnp.ndarray[FLOAT_t, ndim=1] u = <cnp.ndarray[FLOAT_t, ndim=1]> self.u
        cdef cnp.ndarray[FLOAT_t, ndim=2] B_orth = <cnp.ndarray[FLOAT_t, ndim=2]> self.B_orth
        cdef cnp.ndarray[FLOAT_t, ndim=2] X = <cnp.ndarray[FLOAT_t, ndim=2]> self.X
        cdef cnp.ndarray[FLOAT_t, ndim=1] y = <cnp.ndarray[FLOAT_t, ndim=1]> self.y
        cdef cnp.ndarray[FLOAT_t, ndim=1] c = <cnp.ndarray[FLOAT_t, ndim=1]> self.c
        cdef cnp.ndarray[FLOAT_t, ndim=1] B_orth_times_parent_cum = <cnp.ndarray[FLOAT_t, ndim=1]> self.B_orth_times_parent_cum
        cdef cnp.ndarray[FLOAT_t, ndim=2] B = <cnp.ndarray[FLOAT_t, ndim=2]> self.B
        
        cdef unsigned int num_candidates = candidates.shape[0]
        
        cdef unsigned int h
        cdef unsigned int i
        cdef unsigned int j
        cdef FLOAT_t u_end
        cdef FLOAT_t c_end
        cdef FLOAT_t z_end_squared
        cdef unsigned int candidate_idx
        cdef unsigned int last_candidate_idx
        cdef unsigned int last_last_candidate_idx
        cdef unsigned int best_candidate_idx
        cdef FLOAT_t candidate
        cdef FLOAT_t last_candidate
        cdef FLOAT_t best_candidate
        cdef FLOAT_t best_z_end_squared
        cdef FLOAT_t y_cum
        cdef FLOAT_t b_times_parent_cum
        cdef FLOAT_t diff
        cdef FLOAT_t delta_b_squared
        cdef FLOAT_t delta_c_end
        cdef FLOAT_t delta_u_end
        cdef FLOAT_t parent_squared_cum
        cdef FLOAT_t parent_times_y_cum
        cdef FLOAT_t u_dot_c
        cdef FLOAT_t u_dot_u
        cdef FLOAT_t float_tmp
        cdef FLOAT_t delta_b_j
        
        #Compute the initial basis function
        candidate_idx = candidates[0]
        candidate = X[candidate_idx,variable]
        for i in range(self.m):#TODO: Vectorize?
            b[i] = 0
        for i in range(self.m):
            float_tmp = X[i,variable] - candidate
            if float_tmp > 0:
                b[i] = b_parent[i]*float_tmp
            else:
                break
            
        #Put b into the last column of B_orth
        for i in range(self.m):
            B_orth[i,k+1] = b[i]
        
        #Compute the initial covariance column, u (not including the final element)
        u[0:k+1] = np.dot(b,B_orth[:,0:k+1])
        
        #Compute the new last elements of c and u
        c_end = np.dot(b,y)
        u_end = np.dot(b,b)
        
        #Compute the last element of z (the others are identical to c)
        u_dot_c = np.dot(u[0:k+1],c[0:k+1])
        u_dot_u = np.dot(u[0:k+1],u[0:k+1])
        z_end_squared = ((c_end - u_dot_c)**2) / (u_end - u_dot_u)
        
        #Minimizing the norm is actually equivalent to maximizing z_end_squared
        #Store z_end_squared and the current candidate as the best knot choice
        best_z_end_squared = z_end_squared
        best_candidate_idx = candidate_idx
        best_candidate = candidate
        
        #Initialize the accumulators
        last_candidate_idx = 0
        y_cum = y[0]
        B_orth_times_parent_cum[0:k+1] = B_orth[0,0:k+1] * b_parent[0]
        b_times_parent_cum = b[0] * b_parent[0]
        parent_squared_cum = b_parent[0] ** 2
        parent_times_y_cum = b_parent[0] * y[0]
        
        #Now loop over the remaining candidates and update z_end_squared for each, looking for the greatest value
        for i in range(1,num_candidates):
            
            #Update the candidate
            last_last_candidate_idx = last_candidate_idx
            last_candidate_idx = candidate_idx
            last_candidate = candidate
            candidate_idx = candidates[i]
            candidate = X[candidate_idx,variable]
            
            #Update the accumulators and compute delta_b
            diff = last_candidate - candidate
            delta_c_end = 0.0
            
            #What follows is a section of code that has been optimized for speed at the expense of 
            #some readability.  To make it easier to understand this code in the future, I have included a 
            #"simple" block that implements the same math in a more straightforward (but much less efficient) 
            #way.  The (commented out) code between "BEGIN SIMPLE" and "END SIMPLE" should produce the same 
            #output as the code between "BEGIN HYPER-OPTIMIZED" and "END HYPER-OPTIMIZED".
            
            #BEGIN SIMPLE
#            #Calculate delta_b
#            for j  in range(0,last_candidate_idx+1):
#                delta_b[j] = diff
#            for j in range(last_candidate_idx+1,candidate_idx):
#                float_tmp = (X[j,variable] - candidate) * b_parent[j]
#                delta_b[j] = float_tmp
#            
#            #Update u and z_end_squared
#            u[0:k+1] += np.dot(delta_b,B_orth[:,0:k+1])
#            u_end += 2*np.dot(delta_b,b) + np.dot(delta_b, delta_b)
#            
#            #Update c_end
#            c_end += np.dot(delta_b,y)
#            
#            #Update z_end_squared
#            z_end_squared = ((c_end - np.dot(u[0:k+1],c[0:k+1]))**2) / (u_end)
#            
#            #Update b
#            b += delta_b
            #END SIMPLE
            
            #BEGIN HYPER-OPTIMIZED
            delta_b_squared = 0.0
            delta_c_end = 0.0
            delta_u_end = 0.0
            #Update the accumulators
            for j in range(last_last_candidate_idx+1,last_candidate_idx+1):
                y_cum += y[j]
                for h in range(k+1):#TODO: BLAS
                    B_orth_times_parent_cum[h] += B_orth[j,h]*b_parent[j]
                b_times_parent_cum += b[j]*b_parent[j]
                parent_squared_cum += b_parent[j] ** 2
                parent_times_y_cum += b_parent[j] * y[j]
            delta_c_end += diff * parent_times_y_cum
            delta_u_end += 2*diff * b_times_parent_cum
            delta_b_squared = (diff**2)*parent_squared_cum
            
            #Update u and a bunch of other stuff
            for j in range(k+1):
                float_tmp = diff*B_orth_times_parent_cum[j]
                u_dot_c += float_tmp * c[j]
                u_dot_u += 2*u[j]*float_tmp + float_tmp*float_tmp
                u[j] += float_tmp
            for j in range(last_candidate_idx+1,candidate_idx):
                delta_b_j = (X[j,variable] - candidate) * b_parent[j]
                delta_b_squared += delta_b_j**2
                delta_c_end += delta_b_j * y[j]
                delta_u_end += 2*delta_b_j*b[j]
                for h in range(k+1):
                    float_tmp = delta_b_j * B_orth[j,h]
                    u_dot_c += float_tmp * c[h]
                    u_dot_u += 2*u[h]*float_tmp + float_tmp*float_tmp
                    u[h] += float_tmp
                b[j] += delta_b_j
                
            #Update u_end
            delta_u_end += delta_b_squared
            u_end += delta_u_end
            
            #Update c_end
            c_end += delta_c_end
            
            #Update b_times_parent_cum
            b_times_parent_cum += parent_squared_cum * diff
            
            #Compute the new z_end_squared (this is the quantity we're optimizing)
            z_end_squared = ((c_end - u_dot_c)**2) / (u_end - u_dot_u)
            #END HYPER-OPTIMIZED
            
            #Update the best if necessary
            if z_end_squared > best_z_end_squared:
                best_z_end_squared = z_end_squared
                best_candidate_idx = candidate_idx
                best_candidate = candidate
            
        #Compute the mse for the best z_end and set return values
        mse[0] = (self.y_squared - self.c_squared - best_z_end_squared)/self.m
        knot[0] = best_candidate
        knot_idx[0] = best_candidate_idx

    