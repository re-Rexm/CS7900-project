import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import scipy.io as sio
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi
from joblib import Parallel, delayed
import os

# Set number of parallel jobs
n_jobs = -1  # Uses all available cores
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  

def run_single_iteration(run, X, Y, pca_dim, reduced_dim, h1, h2, r, alpha, maxiter):
    print(f'Processing run {run+1}/10...')  # Added progress tracking
    np.random.seed(run)
    
    # PCA - pre-computed outside the loop
    meanX = np.mean(X, axis=1, keepdims=True)
    X_centered = X - meanX
    U, _, _ = svd(X_centered, full_matrices=False)
    X_pca = (U[:, :pca_dim].T @ X_centered)
    
    # Stratified split using scikit-learn
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_pca.T, Y, test_size=0.5, stratify=Y, random_state=run
    )
    X_train, X_test = X_train.T, X_test.T
    
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
    #data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat' 
    #data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat'
    #data_path = r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat'
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat"

    data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\Yale.mat"

    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\colon.mat"
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\lung_discrete.mat"
    #data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\Prostate.mat"

    print('\nStarting experiment...')
    data = sio.loadmat(data_path)
    
    X = data['X'].T
    Y = data['Y'].flatten().astype(int) - 1
    
     # Parameters
    n_run = 8
    pca_dim = 95   #colon =3, lung = 10, prostate = 11, Yale20, else 95
    reduced_dim =40   #colon =3 , lung = 10, prostate = 10, , Yale 20, else 40
    h1, h2 = 2, 10     #colon = 4, 10 , lung = 2, 15, prostate = 3, 20, Ylae 4,20, else 2,10
    r = 2
    alpha = 0.1
    maxiter = 10
    
    
    # Parallel execution of runs
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_iteration)(
            run, X, Y, pca_dim, reduced_dim, h1, h2, r, alpha, maxiter
        ) for run in range(n_run)
    )
    
    # Unpack results
    acc_allma, acc_allma_semi = zip(*results)
    acc_allma = np.array(acc_allma)
    acc_allma_semi = np.array(acc_allma_semi)
    
    # Print Results
    print(f'\nALLDA mean acc: {np.mean(acc_allma):.4f} ± {np.std(acc_allma):.4f}')
    print(f'ALLDA_semi mean acc: {np.mean(acc_allma_semi):.4f} ± {np.std(acc_allma_semi):.4f}')

if __name__ == '__main__':
    main()