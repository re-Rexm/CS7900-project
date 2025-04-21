import numpy as np
from scipy import io

# Load your .mat file
mat_data = io.loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\AR.mat')  # Replace with your file path
X = mat_data['X']
y = mat_data['Y'].flatten()  # Flatten to 1D array

def ar_index(X,Y):
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


