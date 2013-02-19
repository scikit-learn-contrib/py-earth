# distutils: language = c
#cython: cdivision = True
#cython: boundscheck = True
#cython: wraparound = True

from _util cimport gcv, reorderxby, fastr
from _basis cimport Basis, BasisFunction, ConstantBasisFunction, LinearBasisFunction, HingeBasisFunction
from numpy cimport ndarray
from numpy import std, ones, empty, argsort
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
        self.B = ones(shape=(self.m,self.max_terms+1))
        self.sort_tracker = empty(shape=self.m, dtype=int)
        for i in range(self.m):
            self.sort_tracker[i] = i
        self.sorting = empty(shape=self.m, dtype=int)
        self.mwork = empty(shape=self.m, dtype=int)
            
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
        cdef ndarray[FLOAT_t,ndim=2] R = empty(shape=(k+3,k+3))
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
        #TODO: Write this method
        mse[0] = 10.5
        knot_idx[0] = candidates[0]
        knot[0] = self.X[knot_idx[0],variable]
        
    
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
    