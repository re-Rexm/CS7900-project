import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi
from joblib import Parallel, delayed
import os

# Set number of parallel jobs
n_jobs = -1  # Uses all available cores
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def run_single_iteration(run, X_pca, Y, n_class, percent, reduced_dim, h1, h2, r, alpha, maxiter):
    print(f'Processing {percent}% labeled data, run {run+1}/9...')
    np.random.seed(run)
    
    # Find the number of samples in each class
    class_counts = np.bincount(Y)
    min_samples = np.min(class_counts)
    
    # Calculate number of labeled samples based on percentage
    n_labeled = max(1, round(min_samples * (percent/100)))
    n_unlabeled = min_samples - n_labeled
    
    if n_unlabeled < 1:
        raise ValueError("Not enough samples for unlabeled set")
        
    train_idx = []
    test_idx = []
    
    for i in range(n_class):
        idx = np.where(Y == i)[0]
        np.random.shuffle(idx)
        
        if len(idx) >= min_samples:
            train_idx.extend(idx[:n_labeled])
            test_idx.extend(idx[n_labeled:n_labeled+n_unlabeled])
    
    X_train = X_pca[:, train_idx]
    Y_train = Y[train_idx]
    X_test = X_pca[:, test_idx]
    Y_test = Y[test_idx]
    
    # Run ALLDA
    _, W_allma, _ = ALLDA(X_train, Y_train, reduced_dim, h1, r, 1e-5)
    Z_train_allma = W_allma.T @ X_train
    Z_test_allma = W_allma.T @ X_test
    
    # Run ALLDA_semi
    W_semi, _, _, _ = ALLDA_semi(X_train, Y_train, np.hstack([X_train, X_test]),
                                h1, h2, reduced_dim, alpha, maxiter)
    Z_train_semi = W_semi.T @ X_train
    Z_test_semi = W_semi.T @ X_test
    
    # Evaluate using 1-NN
    knn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn1.fit(Z_train_allma.T, Y_train)
    acc_allma = np.mean(knn1.predict(Z_test_allma.T) == Y_test)
    
    knn2 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn2.fit(Z_train_semi.T, Y_train)
    acc_allma_semi = np.mean(knn2.predict(Z_test_semi.T) == Y_test)
    
    return acc_allma, acc_allma_semi

def main():
    # Load data
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat"
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat"
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat"
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat"



    
    print('\nStarting experiment...')
    data = sio.loadmat(data_path)
    
    # Parameters
    X = data['X'].T
    Y = data['Y'].flatten().astype(int) - 1
    n_class = len(np.unique(Y))
    n_run = 9
    pca_dim = 95
    reduced_dim = 40
    h1, h2 = 2, 10
    r = 2
    alpha = 0.1
    maxiter = 10
    label_percents = [20, 30, 40, 50, 60]
    n_perc = len(label_percents)
    
    # Pre-compute PCA for all runs
    meanX = np.mean(X, axis=1, keepdims=True)
    X_centered = X - meanX
    U, _, _ = svd(X_centered, full_matrices=False)
    X_pca = (U[:, :pca_dim].T @ X_centered)
    
    results = {}
    for percent in label_percents:
        print(f'\nProcessing {percent}% labeled data...')
        # Parallel execution of runs for each percentage
        results[percent] = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(run_single_iteration)(
                run, X_pca, Y, n_class, percent, reduced_dim, h1, h2, r, alpha, maxiter
            ) for run in range(n_run)
        )
    
    # Organize results
    acc_allma = np.zeros((n_perc, n_run))
    acc_allma_semi = np.zeros((n_perc, n_run))
    
    for p, percent in enumerate(label_percents):
        acc_allma[p,:], acc_allma_semi[p,:] = zip(*results[percent])
    
    # Print Final Results
    print('\n---- Final Results ----')
    for p in range(n_perc):
        print(f'\nLabeling: {label_percents[p]}%')
        print(f'ALLDA mean acc:      {np.mean(acc_allma[p,:]):.4f} ± {np.std(acc_allma[p,:]):.4f}')
        print(f'ALLDA_semi mean acc: {np.mean(acc_allma_semi[p,:]):.4f} ± {np.std(acc_allma_semi[p,:]):.4f}')

if __name__ == '__main__':
    main()