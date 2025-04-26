import numpy as np
from scipy.linalg import svd
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from ALLDA import ALLDA
from ALLDA_semi import ALLDA_semi
from scipy.io import savemat

# Load COIL20 dataset
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\AR.mat"
data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\COIL20.mat"
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\MSRA25.mat"
#data_path = r"D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\CS7900-project\Data\YaleB.mat"


data = sio.loadmat(data_path)

# Take only first 5 classes, 10 samples per class
X = data['X'].T
Y = data['Y'].flatten().astype(int) - 1  # Convert to 0-based indexing
mask = Y < 5  # First 5 classes
X = X[:, mask]
Y = Y[mask]
X = X[:, ::7]  # Take every 7th sample to reduce size
Y = Y[::7]

# Parameters
pca_dim = 15
reduced_dim = 10
h1 = 2
h2 = 3
r = 2
alpha = 0.1
maxiter = 5

print("Python Results:")
print("-" * 50)
print(f"Input X shape: {X.shape}")
print(f"Input Y shape: {Y.shape}")
print(f"Y unique values: {np.unique(Y)}")
print(f"Samples per class: {np.bincount(Y)}")

# PCA
meanX = np.mean(X, axis=1, keepdims=True)
X_centered = X - meanX
U, s, Vt = svd(X_centered, full_matrices=False)
X_pca = (U[:, :pca_dim].T @ X_centered)

print(f"\nAfter PCA:")
print(f"Mean X first 3 values: {meanX.flatten()[:3]}")
print(f"X_pca shape: {X_pca.shape}")
print(f"X_pca first 3x3 values:\n{X_pca[:3,:3]}")

# Run ALLDA
OBJ, W_allma, S1 = ALLDA(X_pca, Y, reduced_dim, h1, r, 1e-5)
Z_train_allma = W_allma.T @ X_pca

print("\nALLDA results:")
print(f"W_allma shape: {W_allma.shape}")
print(f"W_allma first 3x3 values:\n{W_allma[:3,:3]}")
print(f"Z_train_allma shape: {Z_train_allma.shape}")
print(f"Z_train_allma first 3x3 values:\n{Z_train_allma[:3,:3]}")

# Run ALLDA_semi
W_semi, p, S, Obj = ALLDA_semi(X_pca, Y, X_pca, h1, h2, reduced_dim, alpha, maxiter)
Z_train_semi = W_semi.T @ X_pca

print("\nALLDA_semi results:")
print(f"W_semi shape: {W_semi.shape}")
print(f"W_semi first 3x3 values:\n{W_semi[:3,:3]}")
print(f"Z_train_semi shape: {Z_train_semi.shape}")
print(f"Z_train_semi first 3x3 values:\n{Z_train_semi[:3,:3]}")

matrices_to_save = {
    'X': X,
    'Y': Y,
    'meanX': meanX,
    'X_centered': X_centered,
    'U': U[:, :pca_dim],  # Save only used columns
    'X_pca': X_pca,
    'W_allma': W_allma,
    'S1': S1,
    'Z_train_allma': Z_train_allma,
    'W_semi': W_semi,
    'p': p,
    'S': S,
    'Z_train_semi': Z_train_semi,
    'OBJ_allda': OBJ,
    'Obj_semi': Obj
}

# Add comparison printing function
def print_matrix_info(name, matrix, num_elements=3):
    print(f"\n=== Matrix: {name} ===")
    print(f"Shape: {matrix.shape}")
    if isinstance(matrix, np.ndarray):
        print(f"First {num_elements} elements: ", end='')
        if matrix.size >= num_elements:
            print([f"{x:.6f}" for x in matrix.flatten()[:num_elements]])
        else:
            print([f"{x:.6f}" for x in matrix.flatten()])
        print(f"Min value: {np.min(matrix):.6f}")
        print(f"Max value: {np.max(matrix):.6f}")
        print(f"Mean value: {np.mean(matrix):.6f}")
    else:
        print(f"Type: {type(matrix)}")

# Save results and print detailed information
savemat('coil20_results.mat', matrices_to_save)

print("\nDetailed Matrix Information:")
print("=" * 50)

# Print information for key matrices
for name, matrix in matrices_to_save.items():
    if isinstance(matrix, np.ndarray):
        print_matrix_info(name, matrix)
    elif isinstance(matrix, dict):
        print(f"\n=== Dictionary: {name} ===")
        for k, v in matrix.items():
            if isinstance(v, np.ndarray):
                print_matrix_info(f"{name}.{k}", v)
    else:
        print(f"\n{name}: {type(matrix)}")

print("\nNote: These values can be compared with MATLAB output for verification.")