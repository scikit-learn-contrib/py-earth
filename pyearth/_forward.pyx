# distutils: language = c
#cython: cdivision = True
#cython: boundscheck = True
#cython: wraparound = True

from _util cimport gcv, reorderxby, fastr
from _basis cimport Basis, BasisFunction, ConstantBasisFunction, LinearBasisFunction, HingeBasisFunction
from _choldate cimport cholupdate, choldowndate
from numpy cimport ndarray
from numpy import std, ones, empty, argsort, dot, sqrt as npsqrt
from numpy.linalg import qr
from libc.math cimport sqrt
from libc.math cimport abs
from libc.math cimport log
from libc.math cimport log2

cdef class ForwardPasser:
    
    def __init__(ForwardPasser self, ndarray[FLOAT_t, ndim=2] X, ndarray[FLOAT_t, ndim=1] y, **kwargs):
        cdef unsigned int i
        self.X = X
        self.y = y
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]
        self.endspan = kwargs['endspan'] if 'endspan' in kwargs else -1
        self.minspan = kwargs['minspan'] if 'minspan' in kwargs else -1
        self.endspan_alpha = kwargs['endspan_alpha'] if 'endspan_alpha' in kwargs else -1
        self.minspan_alpha = kwargs['minspan_alpha'] if 'minspan_alpha' in kwargs else -1
        self.max_terms = kwargs['max_terms'] if 'max_terms' in kwargs else -1
        self.max_degree = kwargs['max_degree'] if 'max_degree' in kwargs else 1
        self.thresh = kwargs['thresh'] if 'thresh' in kwargs else 0.001
        self.penalty = kwargs['penalty'] if 'penalty' in kwargs else 3.0
        self.check_every = kwargs['check_every'] if 'check_every' in kwargs else -1
        self.min_search_points = kwargs['min_search_points'] if 'min_search_points' in kwargs else 100
        self.xlabels = kwargs['xlabels'] if 'xlabels' in kwargs else ['x'+str(i) for i in range(self.n)]
        sst = std(self.y)**2
        self.record = ForwardPassRecord(self.m,self.n,self.penalty,sst)
        self.basis = Basis()
        self.B = ones(shape=(self.m,self.max_terms+1), order='F')
        self.sort_tracker = empty(shape=self.m, dtype=int)
        for i in range(self.m):
            self.sort_tracker[i] = i
        self.sorting = empty(shape=self.m, dtype=int)
        self.mwork = empty(shape=self.m, dtype=int)
        self.delta = empty(shape=self.m, dtype=float)
        
            
    cpdef run(ForwardPasser self):
        while True:
            self.next_pair()
            if self.stopCheck():
                break
        
    cdef next_pair(ForwardPasser self):
        cdef unsigned int variable
        cdef unsigned int parent_idx
        cdef unsigned int parent_degree
        cdef unsigned int nonzero_count
        cdef BasisFunction parent
        cdef ndarray[FLOAT_t,ndim=1] candidates_idx
        cdef FLOAT_t knot
        cdef FLOAT_t mse
        cdef unsigned int knot_idx
        cdef FLOAT_t knot_choice
        cdef FLOAT_t mse_choice
        cdef unsigned int knot_idx_choice
        cdef unsigned int parent_choice
        cdef unsigned int variable_choice
        cdef bint first = True
        cdef BasisFunction bf1
        cdef BasisFunction bf2
        cdef unsigned int k = len(self.basis)
        self.R = empty(shape=(k+3,k+3))
        self.u = empty(shape=k+3, dtype=float)
        self.v = empty(shape=k+3, dtype=float)
        
        if self.endspan < 0:
            endspan = round(3 - log2(self.endspan_alpha/self.n))
        
        #Iterate over variables
        for variable in range(self.n):
            
            #Sort the data
            self.sorting[:] = argsort(self.X[:,variable])[::-1] #TODO: eliminate Python call / data copy
            reorderxby(self.X,self.B,self.y,self.sorting,self.sort_tracker)
            
            #Iterate over parents
            for parent_idx in range(k):
                parent = self.basis.get(parent)
                if self.max_degree >= 0:
                    parent_degree = parent.degree()
                    if parent_degree >= self.max_degree:
                        continue
                
                #Add the linear term to B
                self.B[:,k] = self.B[:,parent_idx]*self.X[:,parent_idx] #TODO: Optimize
                
                #Calculate the MSE with just the linear term
                mse = fastr(self.B,self.y,k+1) / self.m
                knot_idx = -1
                
                #Find the valid knot candidates
                candidates_idx = parent.valid_knots(self.B[:,parent_idx], self.X[:,variable],variable, self.check_every, self.endspan, self.minspan, self.minspan_alpha, self.n, self.mwork)
                
                #Choose the best candidate (or None)
                if len(candidates_idx) > 1:
                    self.best_knot(parent_idx,variable,candidates_idx,&mse,&knot,&knot_idx)
                
                #TODO: Recalculate the MSE
                if knot_idx >= 0:
                    bf1 = HingeBasisFunction(parent,knot_choice,variable_choice,False)
                    bf1.apply(self.X,self.B[:,k+1])
                    mse = fastr(self.B,self.y,k+2) / self.m
                
                #Update the choices
                if first:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_choice = parent_idx
                    variable_choice = variable
                    first = False
                if mse < mse_choice:
                    knot_choice = knot
                    mse_choice = mse
                    knot_idx_choice = knot_idx
                    parent_choice = parent_idx
                    variable_choice = variable
        
        #Add the new basis functions
        parent = self.basis.get(parent_idx)
        label = self.xlabels[variable_choice]
        if knot_idx_choice == -1: #Selected linear term
            self.basis.append(LinearBasisFunction(parent,variable_choice,label))
        else:
            bf1 = HingeBasisFunction(parent,knot_choice,variable_choice,False,label)
            bf2 = HingeBasisFunction(parent,knot_choice,variable_choice,True,label)
            bf1.apply(self.X,self.B[:,k])
            bf2.apply(self.X,self.B[:,k+1])
            self.basis.append(bf1)
            self.basis.append(bf2)
        
    cdef best_knot(ForwardPasser self, unsigned int parent, unsigned int variable, ndarray[INT_t,ndim=1] candidates, FLOAT_t * mse, FLOAT_t * knot, unsigned int * knot_idx):
        '''
        Find the best knot location (in terms of squared error).
        
        Assumes:
        B[:,k] is the linear term for variable
        X[:,variable] is in decreasing order
        candidates is in increasing order (it is an array of indices into X[:,variable]
        mse is a pointer to the mean squared error of including just the linear term in B[:,k]
        '''
        cdef unsigned int i
        cdef unsigned int k = len(self.basis)
        cdef unsigned int j
        cdef unsigned int num_candidates
        cdef unsigned int candidate_idx
        cdef FLOAT_t candidate
        cdef unsigned int last_candidate_idx
        cdef FLOAT_t last_candidate
        cdef bint bool_tmp
        cdef FLOAT_t float_tmp
        
        #Put the first candidate into B
        candidate_idx = candidates[0]
        candidate = self.X[candidate_idx,variable]
        for i in range(self.m): #TODO: BLAS
            float_tmp = self.X[i,variable] - candidate
            float_tmp = float_tmp if float_tmp > 0 else 0.0
            self.B[i,k+1] = self.B[i,parent]*float_tmp
            
        #Put y into B to form the augmented data matrix
        for i in range(self.m):
            self.B[i,k+2] = self.y[i]#TODO: BLAS
        
        #Get the cholesky factor using QR decomposition
        self.R[:] = qr(self.B[:,0:k+3],mode='r')
        
        #The lower right corner of the cholesky factor is the norm of the residual
        current_mse = (self.R[k+2,k+2] ** 2) / self.m
        
        #Update the choices
        if current_mse < mse[0]:
            mse[0] = current_mse
            knot_idx[0] = candidate_idx
            knot[0] = candidate
        
        #Initialize the delta vector to 0
        for i in range(self.m):
            self.delta[i] = 0 #TODO: BLAS
        
        #Iterate over remaining candidates
        num_candidates = candidates.shape[0]
        for i in range(1,num_candidates):
            
            #Update the candidate
            last_candidate_idx = candidate_idx
            last_candidate = candidate
            candidate_idx = candidates[i]
            candidate = self.X[candidate_idx,variable]
            
            #Compute the delta vector
            #TODO: BLAS
            #TODO: Optimize
            float_tmp = candidate - last_candidate
            delta_squared = 0.0
            delta_y = 0.0
            for j in range(last_candidate_idx+1):
                self.delta[j] = float_tmp
                self.B[j,k+1] += float_tmp
                delta_squared += float_tmp
                delta_y += float_tmp * self.y[j]
            for j in range(last_candidate_idx+1,candidate_idx):
                float_tmp = self.X[j,variable] - candidate
                self.delta[j] = float_tmp
                self.B[j,k+1] += float_tmp
                delta_squared += float_tmp
                delta_y += float_tmp * self.y[j]
                
            #Compute the u vector
            self.u[0:k+2] = dot(self.B[:,0:k+2],self.delta)
            self.u[k+1] *= 2
            self.u[k+1] += delta_squared
            self.u[k+2] = delta_y
            self.u[:] = npsqrt(self.u)
            
            #Compute the v vector, which is just u with element k+1 zeroed out
            self.v[:] = self.u[:]
            self.v[k+1] = 0
            
            #Update the cholesky factor
            cholupdate(self.R,self.u)
            
            #Downdate the cholesky factor
            choldowndate(self.R,self.v)
            
            #The lower right corner of the cholesky factor is the norm of the residual
            current_mse = (self.R[k+2,k+2] ** 2) / self.m
            
            #Update the choices
            if current_mse < mse[0]:
                mse[0] = current_mse
                knot_idx[0] = candidate_idx
                knot[0] = candidate
    
cdef class ForwardPassRecord:
    def __init__(ForwardPassRecord self, unsigned int num_samples, unsigned int num_variables, FLOAT_t penalty, FLOAT_t sst):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.penalty = penalty
        self.sst = sst
        self.iterations = []
    
    cpdef set_stopping_condition(ForwardPassRecord self, int stopping_condition):
        self.stopping_condition = stopping_condition
    
    def __str__(ForwardPassRecord self):
        result = ''
        result += 'Forward Pass\n'
        result += '-'*80 + '\n'
        result += 'iter\tparent\tvar\tz-knot\tmse\tterms\tcode\tgcv\trsq\tgrsq\n'
        result += '-'*80 + '\n'
        for i, iteration in enumerate(self.iterations):
            result += str(i) + '\t' + str(iteration) + '\t%.3f\t%.3f\t%.3f\n' % (self.gcv(i),self.rsq(i),self.grsq(i) if i>0 else float('-inf'))
        result += 'Stopping Condition: %s\n' % (self.stopping_condition)
        return result
    
    def __len__(ForwardPassRecord self):
        return len(self.iterations)
    
    cpdef append(ForwardPassRecord self, ForwardPassIteration iteration):
        pass
    
    cpdef FLOAT_t mse(ForwardPassRecord self, unsigned int iteration):
        return self.iterations[iteration].mse
    
    cpdef FLOAT_t rsq(ForwardPassRecord self, unsigned int iteration):
        cdef ForwardPassIteration it = self.iterations[iteration]
        cdef FLOAT_t mse = it.mse
        return gcv(mse,it.basisSize,self.num_samples,self.penalty)
    
    cpdef FLOAT_t gcv(ForwardPassRecord self, unsigned int iteration):
        cdef FLOAT_t base = gcv(self.sst,1,self.num_samples,self.penalty)
        cdef FLOAT_t it = self.gcv(iteration)
        return 1 - (it / base)
    
    cpdef FLOAT_t grsq(ForwardPassRecord self, unsigned int iteration):
        cdef FLOAT_t mse0 = self.sst
        cdef FLOAT_t mse = self.mse(iteration)
        return 1 - (mse/mse0)
    
cdef class ForwardPassIteration:
    def __init__(ForwardPassIteration self, unsigned int parent, unsigned int variable, FLOAT_t knot, unsigned int mse, unsigned int size, int code):
        self.parent = parent
        self.variable = variable
        self.knot = knot
        self.mse = mse
        self.size = size
        self.code = code
        
        
    def __str__(self):
        result = '%s\t%s\t%s\t%.4f\t%s\t%s' % (self.selectedParent,self.selectedVariable,'%.4f' % self.selectedKnot if self.selectedKnot is not None else None,self.mse,self.basisSize,self.returnCode)
        return result
    