import numpy as np
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot


def markovNMP(A, B, C, D, t, degree, nmp):
    
    J = np.zeros(shape=(len(t),len(t)))

    for i in range(len(t)):
        for j in range(len(t)):
            if i == j :
                J[i][j] = multi_dot([C,matrix_power(A, degree+nmp-1),B])
            elif j-i == 1:
                J[i][j] = multi_dot([C,matrix_power(A, degree+nmp-2),B])
            for k in range(len(t)):
                if j < i and i - j == k:
                    J[i][j] = multi_dot([C,matrix_power(A, k+degree+nmp-1),B])

    return J



""" 

reference : Gu-Min Jeong, Chong-Ho Choi,
            Iterative learning control for linear discrete time nonminimum phase systems,
            Automatica,
            Volume 38, Issue 2,
            2002,
            Pages 287-291,
            ISSN 0005-1098,

"""