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

def export_sympy(earth_model):
    from sympy import Symbol, Add, Mul, Max, RealNumber, Piecewise, Pow, And
    """
    Exports model in Sympy specific syntax for use with Sympy (and conversion with Sympy Codegen)
    : param earth_model: Trained pyearth model

    """

    def termify(bf, i):
        max_degree = bf.degree()
        parent = bf
        parent_name = parent.__str__()
        bf_dict[parent_name] = i
        terms = []

        bf_index = bf_dict[parent_name]
        
        coef = RealNumber(earth_model.coef_[0][bf_index])
        terms.append(coef)

        for _ in range(max_degree):
            variable_index =  parent.get_variable()
            variable = earth_model.xlabels_[variable_index]
            bf_var = Symbol(variable)
            bf_class = str(type(parent))

            if 'LinearBasisFunction' in bf_class:
                
                term =  bf_var
                terms.append(term)

            elif 'SmoothedHingeBasisFunction' in bf_class:
                
                knot = RealNumber(parent.get_knot())
                knot_minus = RealNumber(parent.get_knot_minus())
                knot_plus = RealNumber(parent.get_knot_plus())
                r = RealNumber(parent.get_r())
                p = RealNumber(parent.get_p())
  
                if parent.get_reverse() == False:
                
                    lower_p = (0, bf_var <= knot_minus)
                    upper_p = (bf_var - knot, bf_var >= knot_plus)
                    left_exp = Mul(p, Pow((bf_var - knot_minus), 2))
                    right_exp = Mul(r, Pow((bf_var - knot_minus), 3))
                    middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
                    middle_exp = (Add(left_exp, right_exp), middle_b)
                    piecewise = Piecewise(lower_p, upper_p, middle_exp)
                    term = piecewise
                    
                    terms.append(term)
  
                elif parent.get_reverse() == True:
#
                    lower_p = (-(bf_var - knot)), (bf_var <= knot_minus)
                    upper_p = (0, bf_var >= knot_plus)
                    left_exp = Mul(p, Pow((bf_var - knot_plus), 2))
                    right_exp = Mul(r, Pow((bf_var - knot_plus), 3))
                    middle_b = And(knot_minus < bf_var, bf_var < knot_plus)
                    middle_exp = (Add(left_exp, right_exp), middle_b)                     
                    piecewise = Piecewise(lower_p, upper_p, middle_exp)
                    term = piecewise
                       
                    terms.append(term)

            elif 'HingeBasisFunction' in bf_class:
            
                
                knot = parent.get_knot()
                if parent.get_reverse() == False:
                    term = Max(0, bf_var - RealNumber(knot))
                    terms.append(term)
 
                elif parent.get_reverse() == True:
                    term = Max(0, RealNumber(knot) - bf_var)
                    terms.append(term)

            elif 'MissingnessBasisFunction' in bf_class:
                print bf_var

            else:
                print bf_var

            parent = parent.get_parent()
        term_list.append(terms)

    # runs through all non-pruned bfs
    bf_dict = {}
    i = 0
    term_list = []
    for bf in earth_model.basis_:
        if not bf.is_pruned():
            termify(bf, i)
            i += 1

    flatten_return = [reduce(lambda a,b: a * b, item) for item in term_list]
    expression = reduce(lambda a,b: a + b, flatten_return)
    
    return expression

