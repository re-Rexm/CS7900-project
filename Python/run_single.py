import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

#1. Load data
#data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat' 
#data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat'
#data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat'
data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat"

data = sio.loadmat(data_path)

X = data['X'].T
Y = data['Y'].flatten().astype(int) - 1

n_class = len(np.unique(Y))
n = X.shape[1]
n_run = 10

acc_allma = np.zeros(n_run)
acc_allma_semi = np.zeros(n_run)

# Parameters
pca_dim = 95
reduced_dim = 40
h1 = 2
h2 = 10
r = 2
alpha = 0.1
maxiter = 10
#print(f"Y shape: {Y.shape}, Y dtype: {Y.dtype}")
#print(f"Y unique values: {np.unique(Y)}")
for run in range(n_run):
    print(f'Run {run+1}/{n_run}...')
    
    # PCA
    meanX = np.mean(X, axis=1, keepdims=True)
    X_centered = X - meanX
    U, s, Vt = svd(X_centered, full_matrices=False)
    X_pca = (U[:, :pca_dim].T @ X_centered)
    
    # 50/50 train test split
    np.random.seed(run)
    
    # Find the number of samples in each class
    class_counts = np.bincount(Y)
    if np.any(class_counts == 0):
        raise ValueError("One or more classes have no samples in the dataset.")
    min_samples = np.min(class_counts)  # smallest class size
    
    # Determine how many samples to take from each class for train/test
    train_per_class = int(np.floor(min_samples/2))
    test_per_class = min_samples - train_per_class

    train_idx = []
    test_idx = []
    for i in range(n_class):
        idx = np.where(Y == i)[0]
        np.random.shuffle(idx)
        
        # Take the determined number of samples
        train_idx.extend(idx[:train_per_class])
        test_idx.extend(idx[train_per_class:train_per_class+test_per_class])
    
    X_train = X_pca[:, train_idx]
    Y_train = Y[train_idx]
    X_test = X_pca[:, test_idx]
    Y_test = Y[test_idx]
    
    # Run ALLDA (assuming you've converted these functions to Python)
    _, W_allma, _ = ALLDA(X_train, Y_train, reduced_dim, h1, r, 1e-5)
    Z_train_allma = W_allma.T @ X_train
    Z_test_allma = W_allma.T @ X_test
    
    # Run ALLDA_semi (assuming you've converted these functions to Python)
    W_semi, _, _, _ = ALLDA_semi(X_train, Y_train, np.hstack([X_train, X_test]), 
                      h1, h2, reduced_dim, alpha, maxiter)
    Z_train_semi = W_semi.T @ X_train
    Z_test_semi = W_semi.T @ X_test
    
    # Evaluate using 1-NN
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(Z_train_allma.T, Y_train)
    pred1 = knn1.predict(Z_test_allma.T)
    acc_allma[run] = np.mean(pred1 == Y_test)
    
    knn2 = KNeighborsClassifier(n_neighbors=1)
    knn2.fit(Z_train_semi.T, Y_train)
    pred2 = knn2.predict(Z_test_semi.T)
    acc_allma_semi[run] = np.mean(pred2 == Y_test)

# Print Results
print(f'\nALLDA mean acc: {np.mean(acc_allma):.4f} ± {np.std(acc_allma):.4f}')
print(f'ALLDA_semi mean acc: {np.mean(acc_allma_semi):.4f} ± {np.std(acc_allma_semi):.4f}')