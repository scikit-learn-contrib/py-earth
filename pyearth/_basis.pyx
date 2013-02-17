# distutils: language = c

cdef class BasisFunction:
    
    def __cinit__(BasisFunction self):
        self.pruned = False
        self.children = []
        self.prunable = True
        
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
            self.child_map[var].append([n])
        
    cpdef prune(self):
        self.pruned = True
        
    cpdef unprune(self):
        self.pruned = False
    
    cpdef knots(BasisFunction self, unsigned int variable):
        
        cdef list children
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
            result.append(self.get_children()[idx].get_knot())
        return result
    
    cpdef unsigned int degree(BasisFunction self):
        return self.parent.degree() + 1
    
    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = True):
        '''
        X - Data matrix
        b - parent vector
        Update b to be the output of self, assuming it was already the output of parent.
        recurse - If False, assume b already contains the result of the parent function.  Otherwise, recurse to compute
                  parent function.
        '''
        
        
    
    
    
    
cdef class ConstantBasisFunction(BasisFunction):
    def __init__(self): #@DuplicatedSignature
        self.prunable = False
    
    cpdef unsigned int degree(ConstantBasisFunction self):
        return 0
    
    cpdef translate(ConstantBasisFunctionself, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse):
        pass
    
    cpdef FLOAT_t scale(ConstantBasisFunctionself, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts):
        return <FLOAT_t> 1.0
    
    cpdef _set_parent(self,BasisFunction parent):
        '''Calls _add_child.'''
        raise NotImplementedError

    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = False):
        '''
        X - Data matrix
        b - parent vector
        Update b to be the output of self, assuming it was already the output of parent.
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
    
    def __init__(self, BasisFunction parent, FLOAT_t knot, unsigned int variable, bint reverse, label=None): #@DuplicatedSignature
        
        self.knot = knot
        self.variable = variable
        self.reverse = reverse
        self.label = label if label is not None else 'x'+str(variable)
        self._set_parent(parent)
    
    cpdef translate(HingeBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, bint recurse):
        self.knot = slopes[self.variable]*self.knot + intercepts[self.variable]
        if slopes[self.variable] < 0:
            self.reverse = not self.reverse
        if recurse:
            self.parent.translate(slopes,intercepts)
            
    cpdef FLOAT_t scale(HingeBasisFunction self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts):
        result = self.parent.scale(slopes,intercepts)
        result /= slopes[self.variable]
        return result
    
    def __str__(self): #@DuplicatedSignature
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
    
    cpdef FLOAT_t get_knot(self):
        return self.knot
    
    cpdef apply(self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=1] b, bint recurse = True):
        '''
        X - Data matrix
        b - parent vector
        Update b to be the output of self, assuming it was already the output of parent.
        recurse - The ConstantBasisFunction is the parent of all BasisFunctions and never has a parent.  
                  Therefore the recurse argument is ignored.  This spares child BasisFunctions from 
                  having to know whether their parents have parents.
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
    

cdef class Basis:
    '''A wrapper that provides functionality related to a set of BasisFunctions with a 
    common ConstantBasisFunction ancestor.  Retains the order in which BasisFunctions are 
    added.'''

    def __init__(Basis self): #@DuplicatedSignature
        self.order = []
    
    cpdef translate(Basis self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts):
        cdef unsigned int n = len(self)
        cdef unsigned int i #@DuplicatedSignature
        for i in range(n):
            self.order[i].translate(slopes,intercepts,False)
        
    cpdef scale(Basis self, np.ndarray[FLOAT_t,ndim=1] slopes, np.ndarray[FLOAT_t,ndim=1] intercepts, np.ndarray[FLOAT_t,ndim=1] beta):
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
    
    cpdef add(Basis self, BasisFunction basis_function):
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
    
    cpdef transform(Basis self, np.ndarray[FLOAT_t,ndim=2] X, np.ndarray[FLOAT_t,ndim=2] B):
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
    