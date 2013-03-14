from _forward import ForwardPasser
from _pruning import PruningPasser
from _util import ascii_table, gcv

import numpy as np

class Earth(object):
    forward_pass_arg_names = set(['endspan','minspan','endspan_alpha','minspan_alpha',
                                  'max_terms','max_degree','thresh','penalty','check_every',
                                  'min_searh_points','xlabels','linvars'])
    pruning_pass_arg_names = set(['penalty'])
    
    def __init__(self, **kwargs):
        self.set_params(**kwargs)
    
    def _pull_forward_args(self, **kwargs):
        result = {}
        for name in self.forward_pass_arg_names:
            if name in kwargs:
                result[name] = kwargs[name]
        return result
    
    def _pull_pruning_args(self, **kwargs):
        result = {}
        for name in self.pruning_pass_arg_names:
            if name in kwargs:
                result[name] = kwargs[name]
        return result
    
    def _pull_unknown_args(self, **kwargs):
        result = {}
        known_args = self.forward_pass_arg_names | self.pruning_pass_arg_names
        for name in kwargs.iterkeys():
            if name not in known_args:
                result[name] = kwargs[name]
        return result
    
    def _scrub_x(self, X, **kwargs):
        no_labels = False
        if 'xlabels' not in kwargs and 'xlabels' not in self.__dict__:
            #Try to get xlabels from input data (for example, if X is a pandas DataFrame)
            try:
                self.xlabels = list(X.columns)
            except AttributeError:
                try:
                    self.xlabels = list(X.design_info.column_names)
                except AttributeError:
                    try:
                        self.xlabels = list(X.dtype.names)
                    except TypeError:
                        no_labels = True
        elif 'xlabels' not in self.__dict__:
            self.xlabels = kwargs['xlabels']
        
        #Convert to internally used data type
        X = np.asarray(X,dtype=np.float64)
        m,n = X.shape
        
        #Make up labels if none were found
        if no_labels:
            self.xlabels = ['x'+str(i) for i in range(n)]
            
        return X
    
    def _scrub(self, X, y, **kwargs):
        #Check whether X is the output of patsy.dmatrices
        if y is None and type(X) is tuple:
            y, X = X
        
        #Handle X separately
        X = self._scrub_x(X, **kwargs)
        
        #Convert y to internally used data type
        y = np.asarray(y,dtype=np.float64)
        y = y.reshape(y.shape[0])
        
        #Make sure dimensions match
        if y.shape[0] != X.shape[0]:
            raise ValueError('X and y do not have compatible dimensions.')
        
        
        return X, y
    
    def set_params(self, **kwargs):
        #Check for unknown arguments
        unknown_args = self._pull_unknown_args(**kwargs)
        if unknown_args:
            msg = 'Unknown arguments: '
            for i, (k, v) in enumerate(unknown_args.iteritems()):
                msg += k+': '+str(v)
                if i < len(unknown_args) - 1:
                    msg += ', '
                else:
                    msg += '.'
            raise ValueError(msg)
        
        #Process forward pass arguments
        self.__dict__.update(self._pull_forward_args(**kwargs))
        
        #Process pruning pass arguments
        self.__dict__.update(self._pull_pruning_args(**kwargs))
    
    def fit(self, X, y = None, xlabels=None, linvars=None):
        #Format and label the data
        if xlabels is not None:
            self.set_params(xlabels=xlabels)
        if linvars is not None:
            self.set_params(linvars=linvars)
        X, y = self._scrub(X,y,**self.__dict__)
        
        #Do the actual work
        self.forward_pass(X, y)
        self.pruning_pass(X, y)
        self.linear_fit(X, y)
        return self
    
    def forward_pass(self, X, y = None, **kwargs):
        #Pull new labels and linear variables if necessary
        if 'xlabels' in kwargs and 'xlabels' not in self.__dict__:
            self.set_params(xlabels=kwargs['xlabels'])
            del kwargs['xlabels']
        if 'linvars' in kwargs and 'linvars' not in self.__dict__:
            self.set_params(linvars=kwargs['linvars'])
            del kwargs['linvars']
        
        #Label and format data
        X, y = self._scrub(X,y,**self.__dict__)
        
        #Check for additional forward pass arguments, and fail if someone tried
        #to use other arguments
        args = self._pull_forward_args(**self.__dict__)
        new_args = self._pull_forward_args(**kwargs)
        if len(new_args) < len(kwargs):
            msg = 'Invalid forward pass arguments: '
            for k, v in kwargs.iteritems():
                if k in new_args:
                    continue
                msg += k+': '+str(v) + ','
            msg = msg[0:-1]+'.'
            raise ValueError(msg)
        args.update(new_args)
        
        #Do the actual work
        forward_passer = ForwardPasser(X, y, **args)
        forward_passer.run()
        self.forward_pass_record_ = forward_passer.trace()
        self.basis_ = forward_passer.get_basis()
        
    def pruning_pass(self, X, y = None, **kwargs):
        #Format data
        X, y = self._scrub(X,y)
        
        #Check for additional pruning arguments and raise ValueError if other arguments are present
        args = self._pull_pruning_args(**self.__dict__)
        new_args = self._pull_pruning_args(**kwargs)
        if len(new_args) < len(kwargs):
            msg = 'Invalid pruning pass arguments: '
            for k, v in kwargs.iteritems():
                if k in new_args:
                    continue
                msg += k+': '+str(v) + ','
            msg = msg[0:-1]+'.'
            raise ValueError(msg)
        args.update(new_args)
        
        #Do the actual work
        pruning_passer = PruningPasser(self.basis_, X, y, **args)
        pruning_passer.run()
        self.pruning_pass_record_ = pruning_passer.trace()
    
    def unprune(self, X, y = None):
        '''Unprune all pruned basis functions and fit coefficients to X and y using the unpruned basis.'''
        for bf in self.basis_:
            bf.unprune()
        del self.pruning_pass_record_
        self.linear_fit(X, y)
    
    def forward_trace(self):
        '''Return information about the forward pass.'''
        try:
            return self.forward_pass_record_
        except AttributeError:
            return None
        
    def pruning_trace(self):
        '''Return information about the pruning pass.'''
        try:
            return self.pruning_pass_record_
        except AttributeError:
            return None
    
    def trace(self):
        '''Return information about the forward and pruning passes.'''
        return EarthTrace(self.forward_trace(),self.pruning_trace())
    
    def summary(self):
        '''Return a string describing the model.'''
        result = ''
        if self.forward_trace() is None:
            result += 'Untrained Earth Model'
            return result
        elif self.pruning_trace() is None:
            result += 'Unpruned Earth Model\n'
        else:
            result += 'Earth Model\n'
        header = ['Basis Function', 'Pruned', 'Coefficient']
        data = []
        i = 0
        for bf in self.basis_:
            data.append([str(bf),'Yes' if bf.is_pruned() else 'No','%g'%self.coef_[i] if not bf.is_pruned() else 'None'])
            if not bf.is_pruned():
                i += 1
        result += ascii_table(header,data)
        if self.pruning_trace() is not None:
            record = self.pruning_trace()
            selection = record.get_selected()
        else:
            record = self.forward_trace()
            selection = len(record) - 1
        result += '\n'
        result += 'MSE: %.4f, GCV: %.4f, RSQ: %.4f, GRSQ: %.4f' % (record.mse(selection), record.gcv(selection), record.rsq(selection), record.grsq(selection))
        return result
    
    def linear_fit(self, X, y = None):
        '''Solve the linear least squares problem to determine the coefficients of the unpruned basis functions.'''
        #Format data
        X, y = self._scrub(X,y)
        
        #Transform into basis space
        B = self.transform(X)
        
        #Solve the linear least squares problem
        self.coef_ = np.linalg.lstsq(B,y)[0]
    
    def predict(self, X):
        '''Predict the response based on the input data X.'''
        X = self._scrub_x(X)
        B = self.transform(X)
        return np.dot(B,self.coef_)
    
    def transform(self, X):
        '''Transform X into the basis space.'''
        X = self._scrub_x(X)
        B = np.empty(shape=(X.shape[0],self.basis_.plen()))
        self.basis_.transform(X,B)
        return B
    
    def get_penalty(self):
        '''Get the penalty parameter being used.  Default is 3.'''
        if 'penalty' in self.__dict__ and self.penalty is not None:
            return self.penalty
        else:
            return 3.0
    
    def score(self, X, y = None):
        '''Calculate the GCV of the model on data X and y.'''
        X, y = self._scrub(X, y)
        y_hat = self.predict(X)
        m, n = X.shape
        residual = y-y_hat
        mse = np.sum(residual**2) / m
        return gcv(mse,self.basis_.plen(),m,self.get_penalty())

    def __str__(self):
        return self.summary()

class EarthTrace(object):
    def __init__(self, forward_trace, pruning_trace):
        self.forward_trace = forward_trace
        self.pruning_trace = pruning_trace
        
    def __str__(self):
        return str(self.forward_trace) + '\n' + str(self.pruning_trace)
    
