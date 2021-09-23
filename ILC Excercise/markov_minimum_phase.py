import numpy as np
from numpy.linalg import matrix_power
from numpy.linalg import multi_dot

def markovMP(A,B,C,D,t): 
    H = np.zeros(shape=(len(t),len(t)))
    
    for row in range(len(t)):
        for col in range(len(t)):
            if row == col :
                H[row][col] = np.dot(C,B)
            for k in range(len(t)):
                if row - col == k:
                    H[row][col] = multi_dot([C,matrix_power(A, k),B])
    
    return H