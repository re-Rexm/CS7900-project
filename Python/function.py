import numpy as np
from scipy import io

def ar_data(X,y):
    # Initialize lists to store selected indices
    selected_indices = []

    # We'll take labels 1 through 100
    target_labels = range(1, 101)

    # For each label, take first 14 occurrences
    for label in target_labels:
        # Find indices where this label occurs
        label_indices = np.where(y == label)[0]

        # Check if we have enough samples (at least 26)
        if len(label_indices) >= 14:
            selected_indices.extend(label_indices[:14])
        else:
            print(f"Warning: Label {label} has only {len(label_indices)} samples (needs 14)")

    # Convert to numpy array
    selected_indices = np.array(selected_indices)

    # Create the subset dataset
    X_subset = X[selected_indices]
    y_subset = y[selected_indices]

    return X_subset, y_subset


def PCA(data):
	data = data - np.mean(data,axis=0) 
	cov = np.dot(data.T,data)
     
	eVals,eVecs = np.linalg.eigh(cov)
	eVecs = np.flip(eVecs,axis=1)
	eVals = np.flip(eVals)	

	return eVals,eVecs

