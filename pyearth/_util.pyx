# distutils: language = c
#cython: boundscheck=True
#cython: wraparound=True

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basisSize, unsigned int dataSize, FLOAT_t penalty):
    return mse / ((1 - ((basisSize + penalty*(basisSize - 1))/dataSize)) ** 2)

