cdef extern from "csteady.c":
     double csteady_rho31(double x)


def steady_rho31(double x):
    return csteady_rho31(x)
    