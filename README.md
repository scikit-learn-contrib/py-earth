py-earth [![Build Status](https://travis-ci.org/jcrudy/py-earth.png?branch=master)](https://travis-ci.org/jcrudy/py-earth?branch=master)
========

A Python implementation of Jerome Friedman's Multivariate Adaptive Regression Splines algorithm, 
in the style of scikit-learn.  I would like to add this code to sklearn in the near future and not maintain 
this separate package.

##Description

The py-earth package implements Multivariate Adaptive Regression Splines using Cython and provides an interface that 
is compatible with scikit-learn's Estimator, Predictor, Transformer, and Model interfaces.  For more information about 
Multivariate Adaptive Regression Splines, see the references below.

##Installation

Make sure you have numpy and scikit-learn installed.  Then do the following:

```
git clone git://github.com/jcrudy/py-earth.git
cd py-earth
sudo python setup.py install
```

##Usage
```python
import numpy
from pyearth import Earth
from matplotlib import pyplot
    
#Create some fake data
numpy.random.seed(0)
m = 1000
n = 10
X = 80*numpy.random.uniform(size=(m,n)) - 40
y = numpy.abs(X[:,6] - 4.0) + 1*numpy.random.normal(size=m)
    
#Fit an Earth model
model = Earth()
model.fit(X,y)
    
#Print the model
print model.trace()
print model.summary()
    
#Plot the model
y_hat = model.predict(X)
pyplot.figure()
pyplot.plot(X[:,6],y,'r.')
pyplot.plot(X[:,6],y_hat,'b.')
pyplot.xlabel('x_6')
pyplot.ylabel('y')
pyplot.title('Simple Earth Example')
pyplot.show()
 ```
 
##Other Implementations

I am aware of the following implementations of Multivariate Adaptive Regression Splines:

1. The R package earth (coded in C by Stephen Millborrow): http://cran.r-project.org/web/packages/earth/index.html
2. The R package mda (coded in Fortran by Trevor Hastie and Robert Tibshirani): http://cran.r-project.org/web/packages/mda/index.html
3. The Orange data mining library for Python (uses the C code from 1): http://orange.biolab.si/
4. The xtal package (uses Fortran code written in 1991 by Jerome Friedman): http://www.ece.umn.edu/users/cherkass/ee4389/xtalpackage.html
5. MARSplines by StatSoft: http://www.statsoft.com/textbook/multivariate-adaptive-regression-splines/
6. MARS by Salford Systems (also uses Friedman's code): http://www.salford-systems.com/products/mars
7. ARESLab (written in Matlab by Gints Jekabsons): http://www.cs.rtu.lv/jekabsons/regression.html

The R package earth was most useful to me in understanding the algorithm, particularly because of Stephen Milborrow's 
thorough and easy to read vignette (http://cran.r-project.org/web/packages/earth/vignettes/earth-notes.pdf).
 
##References

1. Friedman, J. (1991). Multivariate adaptive regression splines. The annals of statistics, 
   19(1), 1–67. http://www.jstor.org/stable/10.2307/2241837
2. Stephen Milborrow. Derived from mda:mars by Trevor Hastie and Rob Tibshirani.
   (2012). earth: Multivariate Adaptive Regression Spline Models. R package
   version 3.2-3. http://CRAN.R-project.org/package=earth
3. Friedman, J. (1993). Fast MARS. Stanford University Department of Statistics, Technical Report No 110. 
   http://statistics.stanford.edu/~ckirby/techreports/LCS/LCS%20110.pdf
4. Friedman, J. (1991). Estimating functions of mixed ordinal and categorical variables using adaptive splines.
   Stanford University Department of Statistics, Technical Report No 108. 
   http://statistics.stanford.edu/~ckirby/techreports/LCS/LCS%20108.pdf
5. Stewart, G.W. Matrix Algorithms, Volume 1: Basic Decompositions. (1998). Society for Industrial and Applied 
   Mathematics.
6. Bjorck, A. Numerical Methods for Least Squares Problems. (1996). Society for Industrial and Applied 
   Mathematics.
7. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning (2nd Edition). (2009).  
   Springer Series in Statistics
8. Golub, G., & Van Loan, C. Matrix Computations (3rd Edition). (1996). Johns Hopkins University Press.
   
References 7, 2, 1, 3, and 4 contain discussions likely to be useful to users of py-earth.  References 1, 2, 6, 5, 
8, 3, and 4 were useful during the implementation process.



   
