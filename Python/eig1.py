# Matlab code to Python
#function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
import numpy as np
def eig1(A, c = None, isMax = None, isSym = None):

#if nargin < 2
#    c = size(A,1);
#    isMax = 1;
#    isSym = 1;

# checking if any argument was passed or not
    if c == None or isMax == None or isSym == None: # only ONE argument was passed
        c = A.shape[0] # number of rows
        isMax = 1
        isSym = 1

#elseif c > size(A,1)
#    c = size(A,1);
#end;

    elif c > A.shape[0]: #ensure c=num of rows of A
        c = A.shape[0]

#if nargin < 3
#    isMax = 1;
#    isSym = 1;
#end;
    # 2 argument passed
    if isMax == None or isSym == None:
        isMax = 1
        isSym = 1


#if nargin < 4
#    isSym = 1;
#end;
#if isSym == 1
#    A = max(A,A');
#end;
    # issym not passed
    if isSym == None:
        isSym = 1
    # symmetric matrix
    if isSym == 1:
        A = np.maximum(A, A.T)

#[v d] = eig(A);
#d = diag(d);
#%d = real(d);

    # eigenvalue and eigenvector
    d, v = np.linalg.eig(A) # already 1D so no nned to diag(d)

#if isMax == 0
#    [d1, idx] = sort(d);
#else
#    [d1, idx] = sort(d,'descend');
#end;

    # sort
    if isMax == 0:
        idx = np.argsort(d) # asending order
    else:
        idx = np.argsort(-d) # decending order

#idx1 = idx(1:c);
#eigval = d(idx1);
#eigvec = v(:,idx1);
#
#eigval_full = d(idx);

    idx1 = idx[:c] # first c index
    eigval = d[idx1] # first c eigenval
    eigvec = v[:, idx1] # same c eigenvec
    eigval_full = d[idx] # all eigenval

    return eigvec, eigval, eigval_full