
cdef extern from "cblas.h":
    double cblas_dnrm2(int N, double *X, int incX)
    void cblas_daxpy(int N, double ALPHA, double *X, int incX, double *Y, int incY)
    void cblas_dcopy(int N, double *X, int incX, double *Y, int incY)
    void cblas_dscal(int N, double ALPHA, double *X, int incX)
    void cblas_dswap(int N, double *X, int incX, double *Y, int incY)