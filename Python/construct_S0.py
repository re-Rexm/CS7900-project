#function  S0 =  construct_S0( idx, h, ni)
import numpy as np
def construct_S0(idx, h, ni):

#% initial a graph in which the weight of nearest
#% neighborhood is one.
#S0 = zeros(ni,ni);

    S0 = np.zeros((idx.shape[0], ni))

#for j = 1:size(idx,1)
#        for k = 1:h
#            if k+1 <= size(idx,2)  % Check bounds
#                S0(j, idx(j,k+1)) = 1/h;
#            end
#        end
    for j in range(idx.shape[0]):
        for k in range(h):
            if k + 1 < idx.shape[1]:  # Check bounds
                S0[j, idx[j, k + 1]] = 1 / h
    return S0
#end
