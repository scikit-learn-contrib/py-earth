
========================================
Multivariate Adaptive Regression Splines
========================================

Multivariate adaptive regression splines, implemented by the Earth class, is flexible 
regression method that automatically searches for interactions and non-linear 
relationships.  Earth models can be thought of as linear models in a higher dimensional 
basis space (specifically, a multivariate truncated power spline basis).  Each term in an 
Earth model is a product of so called "hinge functions".  A hinge function is a function 
that's equal to its argument where that argument is greater than zero and is zero everywhere 
else.
    
The multivariate adaptive regression splines algorithm has two stages.  First, the 
forward pass searches for terms in the truncated power spline basis that locally minimize 
the squared error loss of the training set.  Next, a pruning pass selects a subset of those 
terms that produces a locally minimal generalized cross-validation (GCV) score.  The GCV 
score is not actually based on cross-validation, but rather is meant to approximate a true
cross-validation score by penalizing model complexity.  The final result is a set of terms
that is nonlinear in the original feature space, may include interactions, and is likely to 
generalize well.
    
The Earth class supports dense input only.  Data structures from the pandas and patsy 
modules are supported, but are copied into numpy arrays for computation.  No copy is 
made if the inputs are numpy float64 arrays.  Earth objects can be serialized using the 
pickle module and copied using the copy module.
