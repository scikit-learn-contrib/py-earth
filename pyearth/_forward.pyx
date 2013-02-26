# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = True

from _util cimport reorderxby, fastr
from _basis cimport Basis, BasisFunction, ConstantBasisFunction, LinearBasisFunction, HingeBasisFunction
from _choldate cimport cholupdate, choldowndate
from _record cimport ForwardPassIteration

from libc.math cimport sqrt, abs, log, log2
import numpy as np
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
        sst = np.std(self.y)**2
        self.record = ForwardPassRecord(self.m,self.n,self.penalty,sst)
        self.basis = Basis()
        self.basis.append(ConstantBasisFunction())
        self.B = np.ones(shape=(self.m,self.max_terms+1), order='C',dtype=np.float)
        self.sort_tracker = np.empty(shape=self.m, dtype=int)
        for i in range(self.m):
            self.sort_tracker[i] = i
        self.sorting = np.empty(shape=self.m, dtype=int)
        self.mwork = np.empty(shape=self.m, dtype=int)
        self.delta = np.empty(shape=self.m, dtype=float)
        
    cpdef run(ForwardPasser self):
        while True:
            self.next_pair()
            if self.stop_check():
                break
        
    cpdef Basis get_basis(ForwardPasser self):
        return self.basis
        
    cdef stop_check(ForwardPasser self):
        last = self.record.__len__() - 1
#        if self.record.iterations[last].code == NUMERR:
#            self.record.stopping_condition = NUMDIFF
#            return True
#        if self.record.iterations[last].code != 0:
#            self.record.stopping_condition = NUMDIFF
#            return True
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
        return False
        
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
        self.R = np.empty(shape=(k+3,k+3))
        self.u = np.empty(shape=k+3, dtype=float)
        self.v = np.empty(shape=k+3, dtype=float)
        self.B_cum = np.empty(shape=k+2,dtype=np.float)
        
        cdef cnp.ndarray[FLOAT_t,ndim=2] X = <cnp.ndarray[FLOAT_t,ndim=2]> self.X
        cdef cnp.ndarray[FLOAT_t,ndim=2] B = <cnp.ndarray[FLOAT_t,ndim=2]> self.B
        cdef cnp.ndarray[FLOAT_t,ndim=1] y = <cnp.ndarray[FLOAT_t,ndim=1]> self.y
        
        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha/self.n))
        
        #Iterate over variables
        for variable in range(self.n):
            
            #Sort the data
            self.sorting[:] = np.argsort(X[:,variable])[::-1] #TODO: eliminate Python call / data copy
            reorderxby(X,B,y,self.sorting,self.sort_tracker)
            
            #Iterate over parents
            for parent_idx in range(k):
                parent = self.basis.get(parent_idx)
                if self.max_degree >= 0:
                    parent_degree = parent.degree()
                    if parent_degree >= self.max_degree:
                        continue
                
                #Add the linear term to B
                B[:,k] = B[:,parent_idx]*X[:,variable] #TODO: Optimize
                
                #Calculate the MSE with just the linear term
                mse = fastr(B,y,k+1) / self.m
                knot_idx = -1
                
                #Find the valid knot candidates
                candidates_idx = parent.valid_knots(B[:,parent_idx], X[:,variable], variable, self.check_every, endspan, self.minspan, self.minspan_alpha, self.n, self.mwork)

                #Choose the best candidate (if no candidate is an improvement on the linear term, knot_idx is left as -1)
                if len(candidates_idx) > 1:
                    self.best_knot(parent_idx,variable,candidates_idx,&mse,&knot,&knot_idx)

                #Update the choices
                if first:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    variable_choice = variable
                    first = False
                if mse < mse_choice:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_idx_choice = parent_idx
                    parent_choice = parent
                    variable_choice = variable
        
        #Add the new basis functions
        parent = self.basis.get(parent_idx)
        label = self.xlabels[variable_choice]
        if knot_idx_choice == -1: #Selected linear term
            self.basis.append(LinearBasisFunction(parent,variable_choice,label))
        else:
            bf1 = HingeBasisFunction(parent_choice,knot_choice,knot_idx_choice,variable_choice,False,label)
            bf2 = HingeBasisFunction(parent_choice,knot_choice,knot_idx_choice,variable_choice,True,label)
            bf1.apply(X,B[:,k])
            bf2.apply(X,B[:,k+1])
            self.basis.append(bf1)
            self.basis.append(bf2)
        
        #Update the build record
        self.record.append(ForwardPassIteration(parent_idx_choice,variable_choice,knot_idx_choice,mse_choice,len(self.basis)))
        
        
    cdef best_knot(ForwardPasser self, unsigned int parent, unsigned int variable, cnp.ndarray[INT_t,ndim=1] candidates, FLOAT_t * mse, FLOAT_t * knot, unsigned int * knot_idx):
        '''
        Find the best knot location (in terms of squared error).
        
        Assumes:
        B[:,k] is the linear term for variable
        X[:,variable] is in decreasing order
        candidates is in increasing order (it is an array of indices into X[:,variable]
        mse is a pointer to the mean squared error of including just the linear term in B[:,k]
        '''
        cdef unsigned int h
        cdef unsigned int i
        cdef unsigned int k = len(self.basis)
        cdef unsigned int j
        cdef unsigned int num_candidates
        cdef unsigned int candidate_idx
        cdef FLOAT_t candidate
        cdef unsigned int last_candidate_idx
        cdef unsigned int last_last_candidate_idx
        cdef FLOAT_t last_candidate
        cdef bint bool_tmp
        cdef FLOAT_t float_tmp
        cdef FLOAT_t delta_squared
        cdef FLOAT_t delta_y
        cdef FLOAT_t current_mse
        cdef FLOAT_t y_cum
        cdef FLOAT_t diff
        
        cdef cnp.ndarray[FLOAT_t,ndim=2] X = <cnp.ndarray[FLOAT_t,ndim=2]> self.X
        cdef cnp.ndarray[FLOAT_t,ndim=2] B = <cnp.ndarray[FLOAT_t,ndim=2]> self.B
        cdef cnp.ndarray[FLOAT_t,ndim=2] R = <cnp.ndarray[FLOAT_t,ndim=2]> self.R
        cdef cnp.ndarray[FLOAT_t,ndim=1] y = <cnp.ndarray[FLOAT_t,ndim=1]> self.y
        cdef cnp.ndarray[FLOAT_t,ndim=1] delta = <cnp.ndarray[FLOAT_t,ndim=1]> self.delta
        cdef cnp.ndarray[FLOAT_t,ndim=1] u = <cnp.ndarray[FLOAT_t,ndim=1]> self.u
        cdef cnp.ndarray[FLOAT_t,ndim=1] v = <cnp.ndarray[FLOAT_t,ndim=1]> self.v
        cdef cnp.ndarray[FLOAT_t,ndim=1] B_cum = <cnp.ndarray[FLOAT_t,ndim=1]> self.B_cum
        
        #Put the first candidate into B
        candidate_idx = candidates[0]
        candidate = X[candidate_idx,variable]
        for i in range(self.m):#TODO: Vectorize?
            B[i,k+1] = 0
        for i in range(self.m):
            float_tmp = X[i,variable] - candidate
            if float_tmp > 0:
                B[i,k+1] = B[i,parent]*float_tmp
            else:
                break
            
        #Put y into B to form the augmented data matrix
#        cblas_dcopy(<int>self.m,<double *>self.y.data,1,self.B)
        for i in range(self.m):
            B[i,k+2] = self.y[i]#TODO: BLAS
        
        #Get the cholesky factor using QR decomposition
        R[:] = np.linalg.qr(B[:,0:k+3],mode='r') #TODO: Lapack
        
        #The lower right corner of the cholesky factor is the norm of the residual
        current_mse = (R[k+2,k+2] ** 2) / self.m
        
        #Update the choices
        if current_mse < mse[0]:
            mse[0] = current_mse
            knot_idx[0] = candidate_idx
            knot[0] = candidate
        
        #Initialize the delta vector to 0
        for i in range(self.m):
            delta[i] = 0 #TODO: BLAS

        #Initialize the accumulators
        last_candidate_idx = 0
        y_cum = y[0]
        B_cum[:] = B[0,0:k+2]

        #Iterate over remaining candidates
        num_candidates = candidates.shape[0]
        for i in range(1,num_candidates):
            
            #Update the candidate
            last_last_candidate_idx = last_candidate_idx
            last_candidate_idx = candidate_idx
            last_candidate = candidate
            candidate_idx = candidates[i]
            candidate = X[candidate_idx,variable]
            
            #Compute the delta vector
            #TODO: BLAS
            #TODO: Optimize
            diff = last_candidate - candidate
            delta_squared = 0.0
            delta_y = 0.0
            for j in range(last_last_candidate_idx+1,last_candidate_idx+1):
                y_cum += y[j]
                for h in range(k+2):#TODO: BLAS
                    B_cum[h] += B[j,h]
            delta_y += diff * y_cum
            delta_squared = (diff**2)*(last_candidate_idx+1)
            for j in range(last_candidate_idx+1,candidate_idx):
                float_tmp = X[j,variable] - candidate
                delta[j] = float_tmp
                delta_squared += float_tmp**2
                delta_y += float_tmp * y[j]

            #Compute the u vector
            u[0:k+2] = np.dot(delta[last_candidate_idx+1:candidate_idx],B[last_candidate_idx+1:candidate_idx,0:k+2]) #TODO: BLAS
            u[0:k+2] += diff*B_cum
            B_cum[k+1] += (last_candidate_idx+1) * diff
            B[last_candidate_idx+1:candidate_idx,k+1] += delta[last_candidate_idx+1:candidate_idx] #TODO: BLAS
            float_tmp = u[k+1] * 2
            float_tmp += delta_squared
            float_tmp = sqrt(float_tmp)
            u[k+1] = float_tmp
            u[k+2] = delta_y / float_tmp
            u[0:k+1] /= float_tmp #TODO: BLAS
            
            #Compute the v vector, which is just u with element k+1 zeroed out
            v[:] = u[:]#TODO: BLAS
            v[k+1] = 0
            
            #Update the cholesky factor
            cholupdate(R,u)

            #Downdate the cholesky factor
            choldowndate(R,v)

            #The lower right corner of the cholesky factor is the norm of the residual
            current_mse = (R[k+2,k+2] ** 2) / self.m

            #Update the choices
            if current_mse < mse[0]:
                mse[0] = current_mse
                knot_idx[0] = candidate_idx
                knot[0] = candidate
            

    