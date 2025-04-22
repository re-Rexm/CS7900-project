import numpy as np
from scipy.sparse import spdiags, block_diag
from construct_S import construct_S
from construct_S2 import construct_S2
from construct_S4 import construct_S4
from eig1 import eig1
from L2_distance_1 import l2_distance_1

#function [W,p,S,Obj]  = ALLDA_semi( LX,Y,X,h1,h2,m,alpha,maxiter)
def ALLDA_semi(LX, Y, X, h1, h2, m, alpha, maxiter):
#% LX: labeled data, every column is a sample
#% Y:  label vector
#% X:  all train data  
#% h1: h1 nearest neighborhood in labeled data
#% h2: h2 nearest neighborhood in all data
#% m:  final dimensions
#% alpha: a parameter
#% maxiter: the maximal numbre of iteration
#% written by Steven Wang
#% email: zhengwangml@gmail.com
#
#%% load scale
#[~,n] = size(X);  % n denotes the number of all samples
#c = unique(Y);    
#l = length(Y);    % l denotes the number of labeled samples
#p0 = [];
#
    
    # load
    _, n = X.shape
    c = np.unique(Y)
    l = len(Y)
    p0 = None

#%% initialization
#for i=1:length(c)
#    Xc{i} = LX(:,Y == i);  %  store the each class samples
#    nc(i) = size(Xc{i},2);  %  record the number of each class
#end

    # initialize
    Xc = {}
    nc = {}
    for i in range(len(c)):
        Xc[i] = LX[:, Y == c[i]]  
        nc[i] = Xc[i].shape[1] 

#H = eye(l) - 1/l*ones(l);
#% H = eye(80) - 1/80*ones(80);
#St = LX*H*LX';  % St of labeled data
#invSt = inv(St);
    #print(f"LX shape: {LX.shape}")
    #print(f"Y shape: {Y.shape}")
    H = np.eye(l) - (1 / l * np.ones((l, l)))
    St = LX @ H @ LX .T
    invSt = np.linalg.inv(St)
    #invSt = np.linalg.pinv(St)


#% initialized P similarity matrix of labeled data
#for k = 1 : length(c)
#    Xi = Xc{k};
#    ni = nc(k);   
#    distXi = L2_distance_1(Xi,Xi);
#    [~, idx] = sort(distXi,2);
#    S0{k} = construct_S2(idx, h1, ni);   
#    p0 = blkdiag(p0,S0{k});         
#end

    S0 = {}
    p0 = None
    for k in range(len(c)):
        Xi = Xc[k]
        ni = nc[k]
        distXi = l2_distance_1(Xi, Xi)
        idx = np.argsort(distXi, axis=1)
        S0[k] = construct_S2(idx, h1, ni)
        if p0 is None:
            p0 = S0[k]
        else:
            p0 = block_diag((p0, S0[k])).toarray()

#obj1 = zeros(1,length(c));
#Obj = zeros(1,maxiter);

    obj1 = np.zeros(len(c))
    Obj = np.zeros(maxiter)

#% initialized S similarity matrix of all data
#distXX = L2_distance_1(X,X);
#[~, idx] = sort(distXX,2);
#S0 = construct_S4(idx,h2,n);

    distXX = l2_distance_1(X, X)
    idx = np.argsort(distXX, axis=1)
    S0 = construct_S4(idx, h2, n)

#%% Iteration
#for iter = 1:maxiter
#p = [];
#P = p0;
#S = S0.^2;

    for iter in range(maxiter):
        p = []
        P = p0
        S = S0 ** 2

#% Calculate lapalcian matrix L_p;
#P = (P+P')/2;
#D_p = diag(sum(P));
#L_p = D_p - P;

        P = (P + P.T) / 2
        D_p = np.diag(np.sum(P, axis=1))
        L_p = D_p - P

#% Calculate lapalcian matrix L_s;
#S = (S+S')/2;
#D_s = diag(sum(S));
#L_s = D_s - S;

        S = (S + S.T) / 2
        D_s = np.diag(np.sum(S, axis=1))
        L_s = D_s - S

#% Calculate projection matrix W
#G = invSt*(LX*L_p*LX'+alpha*X*L_s*X'); 
#[W,~,~] = eig1(G, m, 0, 0);
#W = W*diag(1./sqrt(diag(W'*St*W)));

        G = invSt @ (LX @ L_p @ LX.T + alpha * X @ L_s @ X.T)
        # Ensure G is symmetric
        #G = (G + G.T) / 2
        W, _, _ = eig1(G, m, isMax=1, isSym=1)
        W = W @ np.diag(1 / np.sqrt(np.diag(W.T @ St @ W)))

#% Updata matrix P
# for i = 1:length(c)
# Xc{i} = LX(:,Y==i); 
# nc(i) = size(Xc{i},2);
# Xi = Xc{i};
# ni = nc(i);
# distLXx = L2_distance_1(W'*Xi,W'*Xi);
# [~, idx] = sort(distLXx,2);
# PP{i} = construct_S2( idx, h1, ni);
# p = blkdiag(p,PP{i});
# obj1(i) = sum(sum(PP{i}.*distLXx));
#% obj2(i) = sum((sum (dis(:,2:h+1).^(1/(1-2)),2) ).^(1-2));
        p = None
        PP = {}
        for i in range(len(c)):
            Xc[i] = LX[:, Y == i]
            nc[i] = Xc[i].shape[1]
            Xi = Xc[i]
            ni = nc[i]
            distLXx = l2_distance_1(W.T @ Xi, W.T @ Xi)
            idx = np.argsort(distLXx, axis=1)
            PP[i] = construct_S2(idx, h1, ni)
            if p is None:
                p = PP[i]
            else:
                p = block_diag((p, PP[i])).toarray()
            obj1[i] = np.sum(PP[i] * distLXx)

# end
#Obj1 = sum(obj1); 

        Obj1 = np.sum(obj1)

#% Updata matrix S
#distXx = L2_distance_1(W'*X,W'*X);
#[disX, idxX] = sort(distXx,2);
#S =  construct_S(disX + eps, idxX, h2, 2, n);
#Obj2 = alpha*sum(sum(distXx.*(S.^2)));

        distXx = l2_distance_1(W.T @ X, W.T @ X)
        disX = np.sort(distXx, axis=1)
        idxX = np.argsort(distXx, axis=1)
        S = construct_S(disX + np.finfo(float).eps, idxX, h2, 2, n) # eps =2.2204e-16
        Obj2 = alpha * np.sum(distXx * (S**2))

#% Change the variable
# p0 = p;
# S0 = S;

        p0 = p
        S0 = S

#%  Objective function value
#Obj(1,iter) = Obj1 + Obj2;

        Obj[iter] = Obj1 + Obj2

#end
    return W, p, S, Obj

