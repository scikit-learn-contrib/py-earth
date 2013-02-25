# distutils: language = c
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile=True

from libc.math cimport log2
from libc.math cimport log
from libc.math cimport abs
cdef FLOAT_t ZERO_TOL = 1e-16
import numpy as np

cdef class BasisFunction:
    
    def __cinit__(BasisFunction self):
        self.pruned = False
        self.children = []
        self.prunable = True
        self.child_map = {}
        
    cpdef bint has_knot(BasisFunction self):
        return False
        
    cpdef bint is_prunable(BasisFunction self):
        return self.prunable
    
    cpdef bint is_pruned(BasisFunction self):
        return self.pruned
    
    cdef list get_children(BasisFunction self):
        return self.children
    
    cpdef _set_parent(self,BasisFunction parent):
        '''Calls _add_child.'''
        self.parent = parent
        self.parent._add_child(self)
    
    cpdef _add_child(self,BasisFunction child):
        '''Called by _set_parent.'''
        cdef unsigned int n = len(self.children)
        self.children.append(child)
        cdef int var = child.get_variable()
        if var in self.child_map:
            self.child_map[var].append(n)
        else:
            self.child_map[var] = [n]
        
    cpdef prune(self):
        self.pruned = True
        
    cpdef unprune(self):
        self.pruned = False
    
    cpdef knots(BasisFunction self, unsigned int variable):
        
        cdef list children
        cdef BasisFunction child
        if variable in self.child_map:
            children = self.child_map[variable]
        else:
            return []
        cdef unsigned int n = len(children)
        cdef unsigned int i
        cdef list result = []
        cdef int idx
        for i in range(n):
            idx = children[i]
            child = self.get_children()[idx]
            if child.has_knot():
                result.append(child.get_knot())
        return result
    
    cpdef unsigned int degree(BasisFunction self):
        return self.parent.degree() + 1
    
    cpdef apply(self, cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=1] b, bint recurse = True):
        '''
        X - Data matrix
        b - parent vector
        recurse - If False, assume b already contains the result of the parent function.  Otherwise, recurse to compute
                  parent function.
        '''
    
    
    
    cpdef cnp.ndarray[INT_t, ndim=1] valid_knots(BasisFunction self, cnp.ndarray[FLOAT_t,ndim=1] values, cnp.ndarray[FLOAT_t,ndim=1] variable, int variable_idx, unsigned int check_every, int endspan, int minspan, FLOAT_t minspan_alpha, unsigned int n, cnp.ndarray[INT_t,ndim=1] workspace):
        '''
        values - The unsorted values of self in the data set
        variable - The sorted values of variable in the data set
        variable_idx - The index of the variable in the data set
        workspace - An m-vector (where m is the number of samples) used internally
        '''
        cdef unsigned int i
        cdef unsigned int j
        cdef unsigned int k
        cdef unsigned int m = values.shape[0]
        cdef FLOAT_t float_tmp
        cdef INT_t int_tmp
        cdef unsigned int count
        cdef int minspan_
        cdef cnp.ndarray[INT_t, ndim=1] result
        cdef unsigned int num_used
        cdef unsigned int prev
        cdef unsigned int start
        cdef int idx
        cdef int last_idx
        
        #Calculate the used knots
        cdef list used_knots = self.knots(variable_idx)
        used_knots.sort()
        
        #Initialize workspace to 1 where value is nonzero
        count = 0
        for i in range(m):
            if abs(values[i]) > ZERO_TOL:
                workspace[i] = 1
                count += 1
            else:
                workspace[i] = 0
            
        #Calculate minspan
        if minspan < 0:
            minspan_ = <int> (-log2(-(1.0/(n*count))*log(1.0-minspan_alpha)) / 2.5)
        else:
            minspan_ = minspan
            
        #Take out the used points and apply minspan
        num_used = len(used_knots)
        prev = 0
        last_idx = -1
        for i in range(num_used):
            idx = used_knots[i]
            if last_idx == idx:
                continue
            workspace[idx] = 0
            j = idx
            k = 0
            while j > prev + 1 and k < minspan:
                if workspace[j-1]:
                    workspace[j-1] = False
                    k += 1
                j -= 1
            j = idx + 1
            k = 0
            while j < m and k < minspan:
                if workspace[j]:
                    workspace[j] = False
                    k += 1
                j += 1
            prev = idx
            last_idx = idx
        
        #Apply endspan
        i = 0
        j = 0
        while i < endspan:
            if abs(values[j]) > ZERO_TOL:
                workspace[j] = 0
                i += 1
            j += 1
            if j == m:
                break
        i = 0
        j = m - 1
        while i < endspan:
            if abs(values[j]) > ZERO_TOL:
                workspace[j] = 0
                i += 1
            if j == 0:
                break
            j -= 1
        
        #Implement check_every
        int_tmp = 0
        count = 0
        for i in range(m):
            if workspace[i]:
                if (int_tmp % check_every) != 0:
                    workspace[i] = 0
                else:
                    count += 1
                int_tmp += 1
            else:
                int_tmp = 0
        
        #Create result array and return
        result = np.empty(shape=count,dtype=int)
        j = 0
        for i in range(m):
            if workspace[i]:
                result[j] = i
                j += 1
        
        return result
        
        
            
        
        
        
        
    
    
    
cdef class ConstantBasisFunction(BasisFunction):
    def __init__(self): #@DuplicatedSignature
        self.prunable = False
    
    cpdef unsigned int degree(ConstantBasisFunction self):
        return 0
    
    cpdef translate(ConstantBasisFunctionself, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse):
        pass
    
    cpdef FLOAT_t scale(ConstantBasisFunctionself, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts):
        return <FLOAT_t> 1.0
    
    cpdef _set_parent(self,BasisFunction parent):
        '''Calls _add_child.'''
        raise NotImplementedError

    cpdef apply(self, cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=1] b, bint recurse = False):
        '''
        X - Data matrix
        b - parent vector
        recurse - The ConstantBasisFunction is the parent of all BasisFunctions and never has a parent.  
                  Therefore the recurse argument is ignored.  This spares child BasisFunctions from 
                  having to know whether their parents have parents.
        ''' 
        cdef unsigned int i #@DuplicatedSignature
        cdef unsigned int m = len(b)
        for i in range(m):
            b[i] = <FLOAT_t> 1.0
            
    def __str__(self):
        return '(Intercept)'
    
cdef class HingeBasisFunction(BasisFunction):
    
    def __init__(self, BasisFunction parent, FLOAT_t knot, unsigned int knot_idx, unsigned int variable, bint reverse, label=None): #@DuplicatedSignature
        
        self.knot = knot
        self.knot_idx = knot_idx
        self.variable = variable
        self.reverse = reverse
        self.label = label if label is not None else 'x'+str(variable)
        self._set_parent(parent)
    
    cpdef bint has_knot(HingeBasisFunction self):
        return True
    
    cpdef translate(HingeBasisFunction self, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse):
        self.knot = slopes[self.variable]*self.knot + intercepts[self.variable]
        if slopes[self.variable] < 0:
            self.reverse = not self.reverse
        if recurse:
            self.parent.translate(slopes,intercepts)
            
    cpdef FLOAT_t scale(HingeBasisFunction self, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts):
        result = self.parent.scale(slopes,intercepts)
        result /= slopes[self.variable]
        return result
    
    def __str__(self):
        result = ''
        if self.variable is not None:
            if not self.reverse:
                if self.knot >= 0:
                    result = 'h(%s-%G)' % (self.label,self.knot)
                else:
                    result = 'h(%s+%G)' % (self.label,-self.knot)
            else:
                result = 'h(%G-%s)' % (self.knot,self.label)
        parent = str(self.parent) if not self.parent.__class__ is ConstantBasisFunction else ''
        if parent != '':
            result += '*%s' % (str(self.parent),)
        return result
    
    cpdef unsigned int get_variable(self):
        return self.variable
    
    cpdef unsigned int get_knot(self):
        return self.knot_idx
    
    cpdef apply(self, cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=1] b, bint recurse = True):
        '''
        X - Data matrix
        b - parent vector
        recurse - If False, assume b already contains the result of the parent function.  Otherwise, recurse to compute
                  parent function.
        ''' 
        if recurse:
            self.parent.apply(X,b,recurse=True)
        cdef unsigned int i #@DuplicatedSignature
        cdef unsigned int m = len(b) #@DuplicatedSignature
        cdef FLOAT_t tmp
        if self.reverse:
            for i in range(m):
                tmp = self.knot - X[i,self.variable]
                if tmp < 0:
                    tmp = <FLOAT_t> 0.0
                b[i] *= tmp
        else:
            for i in range(m):
                tmp = X[i,self.variable] - self.knot
                if tmp < 0:
                    tmp = <FLOAT_t> 0.0
                b[i] *= tmp

cdef class LinearBasisFunction(BasisFunction):
    def __init__(self, BasisFunction parent, unsigned int variable, label=None): #@DuplicatedSignature
        self.variable = variable
        self.label = label if label is not None else 'x'+str(variable)
        self._set_parent(parent)
        
    cpdef translate(LinearBasisFunctionself, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse):
        pass

    cpdef FLOAT_t scale(LinearBasisFunction self, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts):
        result = self.parent.scale(slopes,intercepts)
        result /= slopes[self.variable]
        return result
    
    def __str__(LinearBasisFunction self):
        return self.label
    
    cpdef unsigned int get_variable(self):
        return self.variable
    
    cpdef apply(self, cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=1] b, bint recurse = True):
        '''
        X - Data matrix
        b - parent vector
        recurse - If False, assume b already contains the result of the parent function.  Otherwise, recurse to compute
                  parent function.
        ''' 
        if recurse:
            self.parent.apply(X,b,recurse=True)
        cdef unsigned int i #@DuplicatedSignature
        cdef unsigned int m = len(b) #@DuplicatedSignature
        for i in range(m):
            b[i] *= X[i,self.variable]
        

cdef class Basis:
    '''A wrapper that provides functionality related to a set of BasisFunctions with a 
    common ConstantBasisFunction ancestor.  Retains the order in which BasisFunctions are 
    added.'''

    def __init__(Basis self): #@DuplicatedSignature
        self.order = []
    
    def piter(Basis self):
        for bf in self.order:
            if not bf.is_pruned():
                yield bf
    
    def __str__(Basis self):
        cdef unsigned int i
        cdef unsigned int n = len(self)
        result = ''
        for i in range(n):
            result += str(self[i])
            result += '\n'
        return result
    
    cpdef translate(Basis self, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts):
        cdef unsigned int n = len(self)
        cdef unsigned int i #@DuplicatedSignature
        for i in range(n):
            self.order[i].translate(slopes,intercepts,False)
        
    cpdef scale(Basis self, cnp.ndarray[FLOAT_t,ndim=1] slopes, cnp.ndarray[FLOAT_t,ndim=1] intercepts, cnp.ndarray[FLOAT_t,ndim=1] beta):
        cdef unsigned int n = len(self) #@DuplicatedSignature
        cdef unsigned int i #@DuplicatedSignature
        cdef unsigned int j = 0
        for i in range(n):
            if self.order[i].is_pruned():
                continue
            beta[j] *= self.order[i].scale(slopes,intercepts)
            j += 1
    
    cpdef BasisFunction get_root(Basis self):
        return self.root
    
    cpdef append(Basis self, BasisFunction basis_function):
        self.order.append(basis_function)
        
    def __iter__(Basis self):
        return self.order.__iter__()
    
    def __len__(Basis self):
        return self.order.__len__()
    
    cpdef BasisFunction get(Basis self, unsigned int i):
        return self.order[i]
    
    def __getitem__(Basis self, unsigned int i):
        return self.get(i)
    
    cpdef unsigned int plen(Basis self):
        cdef unsigned int length = 0
        cdef unsigned int i
        cdef unsigned int n = len(self.order)
        for i in range(n):
            if not self.order[i].is_pruned():
                length += 1
        return length
    
    cpdef transform(Basis self, cnp.ndarray[FLOAT_t,ndim=2] X, cnp.ndarray[FLOAT_t,ndim=2] B):
        cdef unsigned int i #@DuplicatedSignature
        cdef unsigned int n = self.__len__()
        cdef BasisFunction bf
        cdef unsigned int col = 0
        for i in range(n):
            bf = self.order[i]
            if bf.is_pruned():
                continue
            bf.apply(X,B[:,col],recurse=True)
            col += 1
    