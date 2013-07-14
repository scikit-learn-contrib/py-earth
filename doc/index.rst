.. py-earth documentation master file, created by
   sphinx-quickstart on Thu Jul 11 21:44:49 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to py-earth's documentation!
====================================
The py-earth package is a Python implementation of Jerome Friedman's Multivariate Adaptive 
Regression Splines algorithm, in the style of scikit-learn.  For more information about Multivariate 
Adaptive Regression Splines, see the bibliography below.  Py-earth is written in Python and Cython and 
provides an interface that is compatible with scikit-learn's Estimator, Predictor, Transformer, and Model 
interfaces.  Py-earth accommodates input in the form of numpy arrays, pandas DataFrames, patsy DesignMatrix 
objects, or most anything that can be converted into an arrray of floats.  Fitted models can be pickled for 
later use.  


Contents
--------

.. toctree::
   :maxdepth: 2
   
   earth


..
	Indices and tables
	==================
	* :ref:`genindex`
	* :ref:`modindex`
	* :ref:`search`


Bibliography
------------
.. bibliography:: earth_bibliography.bib

References :cite:`Hastie2009`, :cite:`Millborrow2012`, :cite:`Friedman1991`, :cite:`Friedman1993`, 
and :cite:`Friedman1991a` contain discussions likely to be useful to users of py-earth.  
References :cite:`Friedman1991`, :cite:`Millborrow2012`, :cite:`Bjorck1996`, :cite:`Stewart1998`,
:cite:`Golub1996`, :cite:`Friedman1993`, and :cite:`Friedman1991a` were useful during the 
implementation process.

