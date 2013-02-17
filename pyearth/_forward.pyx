# distutils: language = c

from _util cimport gcv
#def gcv(*args):
#    return 1.0

cdef class ForwardPasser:
    
    def __init__(ForwardPasser self, np.ndarray[FLOAT_t, ndim=2] X, np.ndarray[FLOAT_t, ndim=1] y, **kwargs):
        self.X = X
        self.y = y
        self.endspan = kwargs['endspan'] if 'endspan' in kwargs else -1
        self.endspan = kwargs['minspan'] if 'minspan' in kwargs else -1
        self.endspan = kwargs['endspan_alpha'] if 'endspan_alpha' in kwargs else -1
        self.endspan = kwargs['minspan_alpha'] if 'minspan_alpha' in kwargs else -1
        self.endspan = kwargs['max_terms'] if 'max_terms' in kwargs else -1
        self.endspan = kwargs['max_degree'] if 'max_degree' in kwargs else -1
        self.endspan = kwargs['thresh'] if 'thresh' in kwargs else -1
        self.endspan = kwargs['penalty'] if 'penalty' in kwargs else -1
        self.endspan = kwargs['check_every'] if 'check_every' in kwargs else -1
        self.endspan = kwargs['min_search_points'] if 'min_search_points' in kwargs else -1
        self.record = ForwardPassRecord()
        self.basis = Basis()
    
    cpdef run(ForwardPasser self):
        pass
        
    cdef next_pair(ForwardPasser self):
        pass
        
    cdef best_knot(ForwardPasser self):
        pass
    
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
    