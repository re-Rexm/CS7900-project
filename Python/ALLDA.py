#% function [W, S] = local_LDA(X, Y, d, h, r, islocal)
import numpy as np
from scipy.sparse import spdiags, block_diag
from construct_S import construct_S
from construct_S0 import construct_S0
from L2_distance_1 import l2_distance_1
from eig1 import eig1

#function [OBJ,W,S1] = ALLDA(X, Y, d, h, r,perr)

def ALLDA(X, Y, d, h, r, perr):

#% X: data set, every colmun is a sample
#% Y: label vector
#% d: final dimension
#% h: k nearest neighborhood
#% r: paremeter


#c = unique(Y);
#n = length(Y);
#for i=1:length(c)
#    Xc{i} = X(:,Y==i);  %  store the each class samples
#    nc(i) = size(Xc{i},2);  %  record the number of each class
#    ind = find(Y==c(i));
#    WW_w(ind,ind) = length(ind)/n;
#end

    c = np.unique(Y)
    n = len(Y)
    Xc = {}
    nc = []
    WW_w = np.zeros((n, n))

    for i, ci in enumerate(c):
        Xc[i] = X[:, Y == ci]  # Store each class samples
        nc.append(Xc[i].shape[1])  # Record the number of each class
        ind = np.where(Y == ci)[0]
        WW_w[np.ix_(ind, ind)] = len(ind) / n

#H = eye(n) - 1/n*ones(n);
#St = X*H*X';
#invSt = inv(St);
#pre_S = [];
#Obj = 0;
#OBJ = [];
#interval = 1;

    H = np.eye(n) - (1 / n * np.ones((n, n)))
    St = X @ H @ X.T
    invSt = np.linalg.inv(St)
    pre_S = None
    Obj = 0
    OBJ = []
    interval = 1

#% initial a graph S0 for each class, in which K nearest neighborhood
#% of each sample is 1 
#for k = 1 : length(c)
#    Xi = Xc{k};
#    ni = nc(k);   
#    distXi = L2_distance_1(Xi,Xi);
#    [~, idx] = sort(distXi,2);
#    S0{k} = construct_S0( idx, h, ni);   
#    pre_S = blkdiag(pre_S,S0{k});      %    
#end

    for k in range(len(c)):
        Xi = Xc[k]
        ni = nc[k]
        distXi = l2_distance_1(Xi, Xi)
        idx = np.argsort(distXi, axis=1)
        S0_k = construct_S0(idx, h, ni)
        if pre_S is None:
            pre_S = S0_k
        else:
            pre_S = block_diag((pre_S, S0_k)).toarray()

#% Pre_S is the initial graph
#
#pre_S = pre_S - diag(diag(pre_S));
#WW_w = WW_w - diag(diag(WW_w));

    pre_S = pre_S - np.diag(np.diag(pre_S))
    WW_w = WW_w - np.diag(np.diag(WW_w))

#% Iterative calculate the projection W with S;   
#count  = 0;
#while (abs(interval)>=perr&&count<200)
#%  for iter = 1:31
# % updata the weight W 
#    S1 = [];
#    D_w = spdiags(sum(WW_w.*(pre_S),r),0,n,n); % degree matrix
#    L_w = D_w - WW_w.*(pre_S);
#    L_w = (L_w + L_w')/2;
#    Sw = X*L_w*X';
#    Sw = (Sw + Sw')/2;
#    P = invSt*Sw;  
#    [W,~,~] = eig1(P, d, 0, 0);
#    W = W*diag(1./sqrt(diag(W'*St*W)));
#%   updata the graph S    
#    for i=1:length(c)
#    Xc{i} = X(:,Y==i); 
#    nc(i) = size(Xc{i},2);
#    Xi = Xc{i};
#    ni = nc(i);
#    distXi = L2_distance_1(W'*Xi,W'*Xi);
#    [dis, idx] = sort(distXi,2);
#    obj(i) = sum((sum (dis(:,2:h+1).^(1/(1-r)),2) ).^(1-r));
#    S{i} = construct_S( dis+eps,idx, h, r, ni);
#    S1 = blkdiag(S1,S{i});
#    end
#    % convergence code in next line
#    pre_S = S1;
#    pre_S = pre_S.^r;
#    pre_S = (pre_S+pre_S')/2; 
#    interval = Obj - sum(obj);
#    OBJ = [OBJ,sum(obj)];
#    count = count + 1;  
#    Obj =  sum(obj);
#end 

    count = 0
    while abs(interval) >= perr and count < 200:
        S1 = None
        D_w = spdiags(np.sum(WW_w * (pre_S), axis=r-1), 0, n, n).toarray()
        L_w = D_w - WW_w * (pre_S)
        L_w = (L_w + L_w.T) / 2
        Sw = X @ L_w @ X.T
        Sw = (Sw + Sw.T) / 2
        P = invSt @ Sw
        W = eig1(P, c=d, isMax=0, isSym=1)[1]
        diag_vals = np.diag(W.T @ St @ W).copy()  # Make a writable copy
        diag_vals[diag_vals == 0] = np.finfo(float).eps  
        W = W @ np.diag(1 / np.sqrt(diag_vals))

        obj = []
        for i in range(len(c)):
            Xc[i] = X[:, Y == c[i]]
            nc[i] = Xc[i].shape[1]
            Xi = Xc[i]
            ni = nc[i]
            distXi = l2_distance_1(W.T @ Xi, W.T @ Xi)
            dis = np.sort(distXi, axis=1)
            idx = np.argsort(distXi, axis=1)
            obj_i = np.sum(np.sum(dis[:, 1:h + 1] ** (1 / (1 - r)), axis=1) ** (1 - r))
            obj.append(obj_i)
            S_i = construct_S(dis + np.finfo(float).eps, idx, h, r, ni)
            if S1 is None:
                S1 = S_i
            else:
                S1 = block_diag((S1, S_i)).toarray()

        pre_S = S1
        pre_S = pre_S ** r
        pre_S = (pre_S + pre_S.T) / 2
        interval = Obj - sum(obj)
        OBJ.append(sum(obj))
        count += 1
        Obj = sum(obj)

    return OBJ, W, S1




