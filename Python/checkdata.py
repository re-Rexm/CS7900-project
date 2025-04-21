import scipy.io
import numpy as np

def PCA(data):
	data = data - np.mean(data,axis=0) 
	cov = np.dot(data.T,data)
     
	eVals,eVecs = np.linalg.eigh(cov)
	eVecs = np.flip(eVecs,axis=1)
	eVals = np.flip(eVals)	

	return eVals,eVecs

# Load the .mat file
#mat_data = scipy.io.loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\COIL20.mat')
mat_data = scipy.io.loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\AR.mat')
#mat_data = scipy.io.loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\MSRA25.mat')
#mat_data = scipy.io.loadmat(r'D:\0_Work\WSU\CS7900\Project\Rimon_Rojan_Adarsh\Rimon_Rojan_Adarsh\RUN\Data\YaleB.mat')
# Show all variables in the file
print(mat_data.keys())

# Display dimensions and other information
# (assuming common variable names, adjust as needed)
for key in mat_data.keys():
    if not key.startswith('__'):  # Skip metadata
        print(f"{key}: {type(mat_data[key])}, shape: {mat_data[key].shape}")
        
# If you know specific variable names
if 'X' in mat_data:
    print(f"Features shape: {mat_data['X'].shape}")
if 'Y' in mat_data or 'y' in mat_data:
    label_key = 'Y' if 'Y' in mat_data else 'y'
    print(f"Labels shape: {mat_data[label_key].shape}")
    print(f"Unique labels: {np.unique(mat_data[label_key])}")