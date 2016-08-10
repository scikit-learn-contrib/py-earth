
def export_python_function(earth_model):
    """
    Exports model as a pure python function, with no numpy/scipy/sklearn dependencies.
    :param earth_model: Trained pyearth model
    :return: A function that accepts an iterator over examples, and returns an iterator over transformed examples
    """
    i = 0
    accessors = []
    for bf in earth_model.basis_:
        if not bf.is_pruned():
            accessors.append(bf.func_factory(earth_model.coef_[0, i]))
            i += 1

    def func(example_iterator):
        return [sum(accessor(row) for accessor in accessors) for row in example_iterator]
    return func


def export_python_string(earth_model, function_name="model"):
    """
    Exports model as a string that evaluates as python code, with no numpy/scipy/sklearn dependencies.
    :param earth_model: Trained pyearth model
    :param function_name: string, optional, will be the name of the function in the returned string
    :return: string, when executed (either by writing to a file, or using `exec`, will define a python
      function that accepts an iterator over examples, and returns an iterator over transformed examples
    """
    i = 0
    accessors = []
    for bf in earth_model.basis_:
        if not bf.is_pruned():
            accessors.append(bf.func_string_factory(earth_model.coef_[0, i]))
            i += 1

    return """def {:s}(example_iterator):
    accessors = [{:s}]
    for x in example_iterator:
        yield sum(accessor(x) for accessor in accessors)
    """.format(function_name, ",\n\t\t".join(accessors))

def export_sympy_term_expressions(earth_model):
    """
    Construct a list of sympy expressions for all non-pruned terms in the model.
    
    :param earth_model: Trained pyearth model
    :return: a list of sympy expressions representing terms in the model.  These 
      expressions are the symbolic equivalent of the Earth.transform method.

    """
    from sympy import Symbol, Add, Mul, Max, RealNumber, Piecewise, Pow, And
    from ._basis import LinearBasisFunction, HingeBasisFunction, SmoothedHingeBasisFunction, \
          MissingnessBasisFunction, ConstantBasisFunction

    def linear_bf_to_factor(bf):
        return Symbol(bf.label)
    
    def smoothed_hinge_bf_to_factor(bf):
        bf_var = Symbol(bf.label)
        knot = RealNumber(bf.get_knot())
        knot_minus = RealNumber(bf.get_knot_minus())
        knot_plus = RealNumber(bf.get_knot_plus())
        r = RealNumber(bf.get_r())
        p = RealNumber(bf.get_p())
        if bf.get_reverse():
            lower_p = (-(bf_var - knot)), (bf_var <= knot_minus)
            upper_p = (0, bf_var >= knot_plus)
            left_exp = Mul(p, Pow((bf_var - knot_plus), 2))
            right_exp = Mul(r, Pow((bf_var - knot_plus), 3))
            middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
            middle_exp = (Add(left_exp, right_exp), middle_b)                     
            piecewise = Piecewise(lower_p, upper_p, middle_exp)
            factor = piecewise
        else:
            lower_p = (0, bf_var <= knot_minus)
            upper_p = (bf_var - knot, bf_var >= knot_plus)
            left_exp = Mul(p, Pow((bf_var - knot_minus), 2))
            right_exp = Mul(r, Pow((bf_var - knot_minus), 3))
            middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
            middle_exp = (Add(left_exp, right_exp), middle_b)
            piecewise = Piecewise(lower_p, upper_p, middle_exp)
            factor = piecewise
        return factor
    
    def hinge_bf_to_factor(bf):
        bf_var = Symbol(bf.label)
        knot = bf.get_knot()
        if bf.get_reverse():
            factor = Max(0, RealNumber(knot) - bf_var)
        else:
            factor = Max(0, bf_var - RealNumber(knot))
        return factor
    
    def missingness_bf_to_factor(bf):
        # This is the error that should be raised when a user attempts to use functionality
        # that has not yet been implemented.
        # TODO: Implement this
        raise NotImplementedError
    
    def constant_bf_to_factor(bf):
        return RealNumber(1)
    
    bf_to_factor_dispatcher = {LinearBasisFunction: linear_bf_to_factor,
                               SmoothedHingeBasisFunction: smoothed_hinge_bf_to_factor,
                               HingeBasisFunction: hinge_bf_to_factor,
                               MissingnessBasisFunction: missingness_bf_to_factor,
                               ConstantBasisFunction: constant_bf_to_factor}
    
    def bf_to_factor(bf):
        '''
        Convert a BasisFunction to a factor of a term.
        '''
        return bf_to_factor_dispatcher[bf.__class__](bf)
    
    def bf_to_term(bf):
        '''
        Convert a BasisFunction to a term (without coefficient).
        '''
        term = bf_to_factor(bf)
        parent = bf.get_parent()
        if parent is None:
            return term
        else:
            return Mul(term, bf_to_term(parent))
    
    return [bf_to_term(bf) for bf in earth_model.basis_.piter()]


def export_sympy(earth_model, columns=None):
    """
    Constructs a sympy expression or list of sympy expressions from of a trained earth model. 
    
    :param earth_model: Trained pyearth model
    :param columns: The index or indices of the output columns for which expressions are to 
      be constructed.  If an integer is used, a sympy expression is returned.  If indices 
      are given then a list of sympy expressions is returned.  If columns is None, it is treated
      as if columns=0 for models with only one output column or as columns=slice(None) for more than
      one output column.
    :return: a sympy expression or list of sympy expressions equivalent to the Earth.predict method for 
      the selected output columns.

    """
    # Set a sane default for columns
    if columns is None:
        if earth_model.coef_.shape[0] == 1:
            columns = 0
        else:
            columns = slice(None)
    
    # Get basis function terms
    terms = export_sympy_term_expressions(earth_model)
    
    # Handle column choice
    coefs = earth_model.coef_[columns]
    if len(coefs.shape) == 1:
        unwrap = True
        coefs = [coefs]
        n_cols = 1
    else:
        unwrap = False
        n_cols = coefs.shape[0]
    
    # Combine coefficients with terms for each output column
    result = [sum([coefs[i][j] * term for j, term in enumerate(terms)]) for i in range(n_cols)]
    
    if unwrap:
        # Result should be an expression rather than a list of expressions.
        result = result[0]
    return result

        
        
    
    
