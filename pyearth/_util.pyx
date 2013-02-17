# distutils: language = c
#cython: boundscheck=True
#cython: wraparound=True

cpdef inline FLOAT_t gcv(FLOAT_t mse, unsigned int basis_size, unsigned int data_size, FLOAT_t penalty):
    return mse / ((1 - ((basis_size + penalty*(basis_size - 1))/data_size)) ** 2)

