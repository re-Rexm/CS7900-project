#% Given the distance to construct a sparse graph
#function  S =  construct_S( dis, idx, h, r, ni)
import numpy as np
def construct_S(dis, idx, h, r, ni):
#S = zeros(ni,ni);
#a = zeros(ni,ni);
    S = np.zeros((ni, ni))
    a = np.zeros(ni)  
#for j = 1:ni
#    a(j) = sum(dis(j,2:h+1).^(1/(1-r)));
#    for k = 1:h
#        S(j, idx(j,k+1)) = (dis(j,k+1)).^(1/(1-r))/a(j);
#    end
    for i in range(ni):
        a[i] = np.sum(dis[i, 1:h+1] ** (1 / (1 - r)))
        for j in range(h):
            S[i, idx[i, j+1]] = (dis[i, j+1] ** (1 / (1 - r))) / a[i]
#end
    return S