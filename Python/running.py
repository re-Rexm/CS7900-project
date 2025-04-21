import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ALLDA_semi import ALLDA_semi
from ALLDA import ALLDA
import pandas as pd

# Load datasets 
datasets = {
    'Coil20': {'path': 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\data red\Image-data-sets-main\COIL20.mat', 'label_percents': [0.1, 0.2, 0.3, 0.4, 0.5]},
    'AR': {'path': 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\data red\Image-data-sets-main\AR.mat', 'label_percents': [0.2, 0.3, 0.4, 0.5, 0.6]},
    'MSRA25': {'path': 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\data red\Image-data-sets-main\MSRA25.mat', 'label_percents': [0.1, 0.2, 0.3, 0.4, 0.5]},
    'YaleB': {'path': 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\data red\Image-data-sets-main\YaleB.mat', 'label_percents': [0.1, 0.2, 0.3, 0.4, 0.5]},
    'PIE': {'path': 'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\data red\Image-data-sets-main\PIE32x32.mat', 'label_percents': [0.2, 0.3, 0.4, 0.5, 0.6]}
}

results = []

for name, config in datasets.items():
    # Load data
    data = scipy.io.loadmat(config['path'])
    X = data['X']  
    Y = data['Y'].ravel() 
    
    # Split data (50% train, 50% test as in paper)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=8)
    
    for percent in config['label_percents']:
        # Split train into labeled and unlabeled
        X_labeled, X_unlabeled, Y_labeled, _ = train_test_split(
            X_train, Y_train, train_size=percent, random_state=42)
        
        # Parameters (from paper)
        h1 = 10  # k1 in paper
        h2 = 10  # k2 in paper
        m = 30   # reduced dimension
        alpha = 0.1  # from paper
        maxiter = 20  # from convergence plots
        
        # Run ALLDA_semi
        W_semi, _, _, _ = ALLDA_semi(X_labeled.T, Y_labeled, X_train.T, h1, h2, m, alpha, maxiter)
        
        # Run ALLDA (supervised version)
        h = 10  # k in paper
        r = 2    # parameter
        perr = 1e-4  # convergence threshold
        _, W_supervised, _ = ALLDA(X_labeled.T, Y_labeled, m, h, r, perr)
        
        # Evaluate performance
        def evaluate(W, X_train, Y_train, X_test, Y_test):
            # Project data
            X_train_proj = W.T @ X_train.T
            X_test_proj = W.T @ X_test.T
            
            # Train KNN classifier (k=1 as in paper)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_proj.T, Y_train)
            
            # Return accuracy
            return knn.score(X_test_proj.T, Y_test)
        
        acc_semi = evaluate(W_semi, X_train, Y_train, X_test, Y_test)
        acc_supervised = evaluate(W_supervised, X_labeled, Y_labeled, X_test, Y_test)
        
        results.append({
            'Dataset': name,
            'Label Percent': f"{int(percent*100)}%",
            'ALLDA_semi': acc_semi,
            'ALLDA': acc_supervised
        })

# Generate comparison table
df = pd.DataFrame(results)

# Pivot to create table similar to paper
table = df.pivot(index='Dataset', columns='Label Percent', values=['ALLDA_semi', 'ALLDA'])
print(table.to_markdown())