# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile=True

from _util cimport gcv

cdef class Record:
        
    def __getitem__(Record self, int idx):
        return self.iterations[idx]
    
    def __len__(Record self):
        return len(self.iterations)
    
    cpdef append(Record self, Iteration iteration):
        self.iterations.append(iteration)
    
    cpdef FLOAT_t mse(Record self, unsigned int iteration):
        return self.iterations[iteration].get_mse()
    
    cpdef FLOAT_t gcv(Record self, unsigned int iteration):
        cdef Iteration it = self.iterations[iteration]
        cdef FLOAT_t mse = it.mse
        return gcv(mse,it.get_size(),self.num_samples,self.penalty)
    
    cpdef FLOAT_t rsq(Record self, unsigned int iteration):
        cdef FLOAT_t mse0 = self.sst#gcv(self.sst,1,self.num_samples,self.penalty)
        cdef FLOAT_t mse = self.mse(iteration)#gcv(self.mse(iteration):,self.iterations[iteration].get_size(),self.num_samples,self.penalty)#self.gcv(iteration)
        return 1 - (mse / mse0)
    
    cpdef FLOAT_t grsq(Record self, unsigned int iteration):
        cdef FLOAT_t gcv0 = gcv(self.sst,1,self.num_samples,self.penalty)
        cdef FLOAT_t gcv_ = self.gcv(iteration)
        return 1 - (gcv_/gcv0)

cdef class PruningPassRecord(Record):
    def __init__(PruningPassRecord self, unsigned int num_samples, unsigned int num_variables, FLOAT_t penalty, FLOAT_t sst, unsigned int size, FLOAT_t mse):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.penalty = penalty
        self.sst = sst
        self.iterations = [FirstPruningPassIteration(size, mse)]
        
    cpdef set_selected(PruningPassRecord self, unsigned int selected):
        self.selected = selected
    
    def __str__(PruningPassRecord self):
        result = ''
        result += 'Pruning Pass\n'
        result += '-'*80 + '\n'
        result += 'iter\tbf\tterms\tmse\tgcv\trsq\tgrsq\n'
        result += '-'*80 + '\n'
        for i, iteration in enumerate(self.iterations):
            result += str(i) + '\t' + str(iteration) + '\t%.3f\t%.3f\t%.3f\n' % (self.gcv(i),self.rsq(i),self.grsq(i))
        result += 'Selected iteration: ' +  str(self.selected) + '\n'
        return result

cdef class ForwardPassRecord(Record):
    def __init__(ForwardPassRecord self, unsigned int num_samples, unsigned int num_variables, FLOAT_t penalty, FLOAT_t sst):
        self.num_samples = num_samples
        self.num_variables = num_variables
        self.penalty = penalty
        self.sst = sst
        self.iterations = [FirstForwardPassIteration(self.sst)]
        
    cpdef set_stopping_condition(ForwardPassRecord self, int stopping_condition):
        self.stopping_condition = stopping_condition
    
    def __str__(ForwardPassRecord self):
        result = ''
        result += 'Forward Pass\n'
        result += '-'*80 + '\n'
        result += 'iter\tparent\tvar\tknot\tmse\tterms\tgcv\trsq\tgrsq\n'
        result += '-'*80 + '\n'
        for i, iteration in enumerate(self.iterations):
            result += str(i) + '\t' + str(iteration) + '\t%.3f\t%.3f\t%.3f\n' % (self.gcv(i),self.rsq(i),self.grsq(i) if i>0 else float('-inf'))
        result += 'Stopping Condition: %s\n' % (self.stopping_condition)
        return result

cdef class Iteration:
    
    cpdef FLOAT_t get_mse(Iteration self):
        return self.mse
    
    cpdef unsigned int get_size(Iteration self):
        return self.size

cdef class PruningPassIteration(Iteration):
    def __init__(PruningPassIteration self, unsigned int pruned, unsigned int size, FLOAT_t mse):
        self.pruned = pruned
        self.size = size
        self.mse = mse
        
    def __str__(PruningPassIteration self):
        result = '%s\t%s\t%s' % (str(self.pruned),self.size,'%.2f' % self.mse if self.mse is not None else None)
        return result
    
cdef class FirstPruningPassIteration(PruningPassIteration):
    def __init__(PruningPassIteration self, unsigned int size, FLOAT_t mse):
        self.size = size
        self.mse = mse

    def __str__(PruningPassIteration self):
        result = '%s\t%s\t%s' % ('-',self.size,'%.2f' % self.mse if self.mse is not None else None)
        return result
    
cdef class ForwardPassIteration(Iteration):
    def __init__(ForwardPassIteration self, unsigned int parent, unsigned int variable, int knot, FLOAT_t mse, unsigned int size):
        self.parent = parent
        self.variable = variable
        self.knot = knot
        self.mse = mse
        self.size = size
        
    def __str__(self):
        result = '%d\t%d\t%d\t%4f\t%d' % (self.parent,self.variable,self.knot,self.mse,self.size)
        return result
    
cdef class FirstForwardPassIteration(ForwardPassIteration):
    def __init__(FirstForwardPassIteration self, FLOAT_t mse):
        self.mse = mse
        
    cpdef unsigned int get_size(FirstForwardPassIteration self):
        return 1
        
    def __str__(self):
        result = '%s\t%s\t%s\t%4f\t%s' % ('-','-','-',self.mse,1)
        return result
    