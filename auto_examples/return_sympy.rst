

.. _sphx_glr_auto_examples_return_sympy.py:


=====================================================
Exporting a fitted Earth models as a sympy expression
=====================================================

A simple example returning a sympy expression describing the fit of a sine function computed by Earth.





.. rst-class:: sphx-glr-script-out

 Out::

      Earth Model
    ----------------------------------------------------
    Basis Function                 Pruned  Coefficient  
    ----------------------------------------------------
    (Intercept)                    No      129.933      
    h(x6+36.8964)                  No      -159.119     
    h(-36.8964-x6)                 Yes     None         
    h(x6+33.7644)*h(x6+36.8964)    No      -31.2137     
    h(-33.7644-x6)*h(x6+36.8964)   No      55.2555      
    h(x6+38.8328)*h(-36.8964-x6)   No      -26.2926     
    h(-38.8328-x6)*h(-36.8964-x6)  No      37.8735      
    h(x6+32.7734)                  No      294.063      
    h(-32.7734-x6)                 No      -102.518     
    h(x6+30.8809)*h(x6+32.7734)    Yes     None         
    h(-30.8809-x6)*h(x6+32.7734)   No      -29.2847     
    h(x6+34.5078)*h(x6+36.8964)    No      21.394       
    h(-34.5078-x6)*h(x6+36.8964)   Yes     None         
    h(x6+37.4374)*h(-32.7734-x6)   No      -15.3682     
    h(-37.4374-x6)*h(-32.7734-x6)  Yes     None         
    ----------------------------------------------------
    MSE: 95.2350, GCV: 100.3872, RSQ: 0.9812, GRSQ: 0.9802
    Resulting sympy expression:
    37.8735010361507*Max(0, -x6 - 38.8328464302324)*Max(0, -x6 - 36.8964359919483) - 26.2925536811075*Max(0, -x6 - 36.8964359919483)*Max(0, x6 + 38.8328464302324) + 55.2555446285282*Max(0, -x6 - 33.7643681543787)*Max(0, x6 + 36.8964359919483) - 15.3681937683734*Max(0, -x6 - 32.7734309073815)*Max(0, x6 + 37.437361535298) - 102.518379033784*Max(0, -x6 - 32.7734309073815) - 29.2847044482295*Max(0, -x6 - 30.8809078363996)*Max(0, x6 + 32.7734309073815) + 294.063440865261*Max(0, x6 + 32.7734309073815) - 31.2137142245201*Max(0, x6 + 33.7643681543787)*Max(0, x6 + 36.8964359919483) + 21.3939946424227*Max(0, x6 + 34.5077812314211)*Max(0, x6 + 36.8964359919483) - 159.11865251946*Max(0, x6 + 36.8964359919483) + 129.932616947174




|


.. code-block:: python


    import numpy
    from pyearth import Earth
    from pyearth import export

    # Create some fake data
    numpy.random.seed(2)
    m = 1000
    n = 10
    X = 10 * numpy.random.uniform(size=(m, n)) - 40
    y = 100 * \
        (numpy.sin((X[:, 6])) - 4.0) + \
        10 * numpy.random.normal(size=m)

    # Fit an Earth model
    model = Earth(max_degree=2, minspan_alpha=.5, verbose=False)
    model.fit(X, y)

    print(model.summary())

    #return sympy expression 
    print('Resulting sympy expression:')
    print(export.export_sympy(model))


**Total running time of the script:**
(0 minutes 1.310 seconds)



.. container:: sphx-glr-download

    **Download Python source code:** :download:`return_sympy.py <return_sympy.py>`


.. container:: sphx-glr-download

    **Download IPython notebook:** :download:`return_sympy.ipynb <return_sympy.ipynb>`
