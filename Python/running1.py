from scipy.io import loadmat
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi
import numpy as np

# Load data
data = loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\COIL20.mat')
X = data['fea'].T   # Transpose to make each column a sample
Y = data['gnd'].flatten()


def PCA(data):
    data = data - np.mean(data, axis=0)
    cov = np.dot(data.T, data)
    eVals, eVecs = np.linalg.eigh(cov)
    eVecs = np.flip(eVecs, axis=1)
    eVals = np.flip(eVals)
    return eVals, eVecs
    

# Parameters
d = 30       # final dimension
h = 5        # nearest neighbors
r = 0.5      # parameter for ALLDA
perr = 1e-3  # convergence tolerance
alpha = 0.1
maxiter = 30

# ALLDA (supervised)
OBJ_allda, W_allda, S1_allda = ALLDA(X, Y, d, h, r, perr)

# Create labeled subset for semi-supervised
def get_labeled_subset(X, Y, percent):
    labels = np.unique(Y)
    X_labeled, Y_labeled = [], []
    for label in labels:
        idx = np.where(Y == label)[0]
        n_label = int(len(idx) * percent)
        idx_labeled = np.random.choice(idx, n_label, replace=False)
        X_labeled.append(X[:, idx_labeled])
        Y_labeled.extend(Y[idx_labeled])
    return np.hstack(X_labeled), np.array(Y_labeled)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def evaluate(W, X_train, y_train, X_test, y_test):
    X_proj_train = W.T @ X_train
    X_proj_test = W.T @ X_test
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_proj_train.T, y_train)
    return accuracy_score(y_test, clf.predict(X_proj_test.T))

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)

results = {}
for percent in [0.1, 0.2, 0.3, 0.4]:
    LX, Y_l = get_labeled_subset(X_train, y_train, percent)
    W_semi, p, S, Obj = ALLDA_semi(LX, Y_l, X_train, h, h, d, alpha, maxiter)
    acc = evaluate(W_semi, X_train, y_train, X_test, y_test)
    results[f"ALLDA_semi_{int(percent*100)}%"] = acc

# Print table
print(f"{'Label Rate':<12} | {'Accuracy':<8}")
print("-" * 25)
for k, v in results.items():
    print(f"{k[-4:]:<12} | {v*100:.2f}%")
