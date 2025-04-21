# MATLAB CODE
#function d = L2_distance_1(a,b)

#compute squared Euclidean distance
#||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B

#if (size(a,1) == 1)
#  a = [a; zeros(1,size(a,2))]; 
#  b = [b; zeros(1,size(b,2))]; 
#end
#
#aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
#d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;
#
#d = real(d);
#d = max(d,0);

# Python equivalent
import numpy as np
def l2_distance_1(a, b):
    # Check if single row), add a row of zeros
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])
    
    # Compute squared Euclidean distance
    aa = np.sum(a * a, axis=0)  # ||A||^2 for each column
    bb = np.sum(b * b, axis=0)  # ||B||^2 for each column
    ab = a.T @ b                # A'*B
    
    # ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
    d = aa[:, np.newaxis] + bb[np.newaxis, :] - 2 * ab
    
    # Change to real and positive number 
    d = np.real(d)
    d = np.maximum(d, 0)
    
    return d