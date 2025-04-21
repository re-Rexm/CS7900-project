#function  S0 =  construct_S4( idx, h,n)
import numpy as np
def construct_S4(idx, h, n):

#% initial a graph in which the weight of nearest
#% neighborhood is one.
#S0 = zeros(n,n);
    S0 = np.zeros((n, n))

#for j = 1:n
#    for k = 1:h
#        S0(j, idx(j,k+1)) = 1/h;
#    end
    for i in range(n):
        for j in range(h):
            S0[i, idx[i, j + 1]] = 1 / h
#end
    return S0