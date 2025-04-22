import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# 1. Load data
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat"
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat"
data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat"
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat"

data = sio.loadmat(data_path)

X = data['X'].T
Y = data['Y'].flatten().astype(int) - 1  # Convert to 0-based indexing

n_class = len(np.unique(Y))
n = X.shape[1]
n_run = 10

# Parameters
pca_dim = 95
reduced_dim = 40
h1 = 2
h2 = 10
r = 2
alpha = 0.1
maxiter = 10
label_percents = [10, 20, 30, 40, 50]
# label_percents = [20, 30, 40, 50, 60]  # For AR dataset only
n_perc = len(label_percents)

acc_allma = np.zeros((n_perc, n_run))        # For supervised ALLDA
acc_allma_semi = np.zeros((n_perc, n_run))   # For semi-supervised ALLDA_semi

for p in range(n_perc):
    percent = label_percents[p]
    print(f'\nRunning for {percent}% labeled data...')

    for run in range(n_run):
        print(f'Run {run+1}/{n_run}...')

        # 2. PCA
        meanX = np.mean(X, axis=1, keepdims=True)
        X_centered = X - meanX
        U, s, Vt = svd(X_centered, full_matrices=False)
        X_pca = (U[:, :pca_dim].T @ X_centered)

        # 3. Balanced Train/Test Split with Percentage Labeling
        np.random.seed(run)  # for reproducibility

        # Find the number of samples in each class
        class_counts = np.bincount(Y)
        if np.any(class_counts == 0):
            raise ValueError("One or more classes have no samples in the dataset.")
        min_samples = np.min(class_counts)  # smallest class size

        # Calculate number of labeled samples based on percentage
        n_labeled = max(1, round(min_samples * (percent/100)))  # Ensure at least one labeled sample
        n_unlabeled = min_samples - n_labeled

        # Error check for unlabeled samples
        if n_unlabeled < 1:
            raise ValueError("Not enough samples for unlabeled set with current percentage")

        train_idx = []
        test_idx = []

        for i in range(n_class):
            idx = np.where(Y == i)[0]
            np.random.shuffle(idx)  # shuffle

            # Take the determined number of samples
            if len(idx) >= min_samples:
                train_idx.extend(idx[:n_labeled])
                test_idx.extend(idx[n_labeled:n_labeled+n_unlabeled])
            else:
                # For classes smaller than min_samples (shouldn't happen due to earlier check)
                print(f'Warning: Class {i} has fewer samples ({len(idx)}) than min_samples ({min_samples})')
                continue

        X_train = X_pca[:, train_idx]
        Y_train = Y[train_idx]
        X_test = X_pca[:, test_idx]
        Y_test = Y[test_idx]

        # 4. Run ALLDA
        _, W_allma, _ = ALLDA(X_train, Y_train, reduced_dim, h1, r, 1e-5)
        Z_train_allma = W_allma.T @ X_train
        Z_test_allma = W_allma.T @ X_test

        # 5. Run ALLDA_semi
        W_semi, _, _, _ = ALLDA_semi(X_train, Y_train, np.hstack([X_train, X_test]), 
                          h1, h2, reduced_dim, alpha, maxiter)
        Z_train_semi = W_semi.T @ X_train
        Z_test_semi = W_semi.T @ X_test

        # 6. Evaluate using 1-NN
        knn1 = KNeighborsClassifier(n_neighbors=1)
        knn1.fit(Z_train_allma.T, Y_train)
        pred1 = knn1.predict(Z_test_allma.T)
        acc_allma[p, run] = np.mean(pred1 == Y_test)

        knn2 = KNeighborsClassifier(n_neighbors=1)
        knn2.fit(Z_train_semi.T, Y_train)
        pred2 = knn2.predict(Z_test_semi.T)
        acc_allma_semi[p, run] = np.mean(pred2 == Y_test)

# Print Final Results
print('\n---- Final Results ----')
for p in range(n_perc):
    print(f'\nLabeling: {label_percents[p]}%')
    print(f'ALLDA mean acc:       {np.mean(acc_allma[p,:]):.4f} ± {np.std(acc_allma[p,:]):.4f}')
    print(f'ALLDA_semi mean acc:  {np.mean(acc_allma_semi[p,:]):.4f} ± {np.std(acc_allma_semi[p,:]):.4f}')