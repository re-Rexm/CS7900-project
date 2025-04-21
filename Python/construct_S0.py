#function  S0 =  construct_S0( idx, h, ni)
import numpy as np
def construct_S0(idx, h, ni):

#% initial a graph in which the weight of nearest
#% neighborhood is one.
#S0 = zeros(ni,ni);

    S0 = np.zeros((ni, ni))

#for j = 1:ni
#    for k = 1:h
#        S0(j, idx(j,k+1)) = 1/h;
#    end
    for i in range(ni):
        for j in range(h):
            S0[i, idx[i, j+1]] = 1 / h
    
    return S0
#end
