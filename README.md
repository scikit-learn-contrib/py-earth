py-earth
========

A Python implementation of Jerome Friedman's MARS algorithm, in the style of scikit-learn.  Long term, I would like to add this to sklearn and not maintain this separate package.


##Usage

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
    print model
    
    #Plot the model
    y_hat = model.predict(X)
    pyplot.figure()
    pyplot.plot(X[:,6],y,'r.')
    pyplot.plot(X[:,6],y_hat,'b.')
    pyplot.show()
