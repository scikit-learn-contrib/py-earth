import numpy
from pyearth import Earth
from matplotlib import pyplot

# Create some fake data
numpy.random.seed(2)
m = 10000
n = 10
X = 20 * numpy.random.uniform(size=(m, n)) - 10
y = 10*numpy.sin(X[:, 6])  + \
    0.25*numpy.random.normal(size=m)
    
# Compute the known true derivative with respect to the predictive variable
y_prime = 10*numpy.cos(X[:, 6])

# Fit an Earth model
model = Earth(max_degree=2, minspan_alpha=.5, smooth=True)
model.fit(X, y)

# Print the model
print model.trace()
print model.summary()

# Get the predicted values and derivatives
y_hat = model.predict(X)
y_prime_hat = model.predict_deriv(X)

# Plot true and predicted function values and derivatives for the predictive variable
pyplot.subplot(211)
pyplot.plot(X[:, 6], y, 'r.')
pyplot.plot(X[:, 6], y_hat, 'b.')
pyplot.ylabel('function')
pyplot.subplot(212)
pyplot.plot(X[:, 6], y_prime, 'r.')
pyplot.plot(X[:, 6], y_prime_hat[:,6], 'b.')
pyplot.ylabel('derivative')
pyplot.show()
