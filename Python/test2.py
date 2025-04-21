import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# --- Step 1: Load Data ---
data = loadmat(r'D:/0_Work/WSU/CS7900/Project/Rimon_Rojan_Adarsh/Rimon_Rojan_Adarsh/RUN/Data/COIL20.mat')
X = data['X'].astype(float).T  # shape (features, samples)
Y = data['Y'].flatten()        # shape (samples,)

# --- Step 2: Split 50/50 Train/Test ---
X = X.T  # shape (samples, features) for sklearn
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=42)

# --- Step 3: PCA ---
def PCA(data):
    data -= np.mean(data, axis=0)
    cov = np.dot(data.T, data)
    eVals, eVecs = np.linalg.eigh(cov)
    eVecs = np.flip(eVecs, axis=1)
    eVals = np.flip(eVals)
    return eVals, eVecs

eVals, eVecs = PCA(X)
n_components = 95  # or use explained variance threshold
X_train_pca = X_train @ eVecs[:, :n_components]
X_test_pca = X_test @ eVecs[:, :n_components]

# --- Step 4: Run ALLDA ---
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi

h = 5
r = 2
perr = 1e-3
OBJ, W_allda, S1 = ALLDA(X_train_pca.T, Y_train, d=30, h=h, r=r, perr=perr)
LX = X_train_pca.T
X_all = np.hstack((X_train_pca.T, X_test_pca.T))
Y_semi = Y_train
W_semi, p, S, Obj = ALLDA_semi(LX, Y_semi, X_all, h1=5, h2=5, m=30, alpha=0.01, maxiter=10)

# --- Step 5: Evaluate Mean Recognition Accuracy ---
def evaluate(W, X_train, Y_train, X_test, Y_test):
    clf = KNeighborsClassifier(n_neighbors=1)
    X_train_proj = (W.T @ X_train.T).T
    X_test_proj = (W.T @ X_test.T).T
    clf.fit(X_train_proj, Y_train)
    acc = clf.score(X_test_proj, Y_test)
    return acc

n_trials = 10
accs_allda = []
accs_allda_semi = []

for i in range(n_trials):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y, random_state=i)
    eVals, eVecs = PCA(X_train)
    X_train_pca = X_train @ eVecs[:, :n_components]
    X_test_pca = X_test @ eVecs[:, :n_components]

    OBJ, W_allda, _ = ALLDA(X_train_pca.T, Y_train, d=30, h=h, r=r, perr=perr)
    LX = X_train_pca.T
    X_all = np.hstack((X_train_pca.T, X_test_pca.T))
    W_semi, _, _, _ = ALLDA_semi(LX, Y_train, X_all, h1=5, h2=5, m=30, alpha=0.01, maxiter=10)

    accs_allda.append(evaluate(W_allda, X_train_pca, Y_train, X_test_pca, Y_test))
    accs_allda_semi.append(evaluate(W_semi, X_train_pca, Y_train, X_test_pca, Y_test))

# Results
mean_allda = np.mean(accs_allda) * 100
std_allda = np.std(accs_allda) * 100
mean_semi = np.mean(accs_allda_semi) * 100
std_semi = np.std(accs_allda_semi) * 100

print(f"ALLDA Accuracy: {mean_allda:.2f}% ± {std_allda:.2f}%")
print(f"ALLDA-semi Accuracy: {mean_semi:.2f}% ± {std_semi:.2f}%")
