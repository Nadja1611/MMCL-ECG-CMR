import numpy as np
from scipy.signal import savgol_filter

# Function to remove baseline drift using asymmetric least square smoothing
def remove_baseline_drift(ecg_signal, lam=1e6, p=0.01, niter=10):
    L = len(ecg_signal)
    D = np.diff(np.eye(L), 2)
    D = np.dot(D.T, D)
    w = np.ones(L)
    
    for _ in range(niter):
        W = np.diag(w)
        Z = np.linalg.inv(W + lam * D).dot(W).dot(ecg_signal)
        w = p * (ecg_signal > Z) + (1 - p) * (ecg_signal < Z)
        
    return ecg_signal - Z

# Function to normalize the ECG leads
def normalize_lead(ecg_lead):
    return (ecg_lead - np.mean(ecg_lead)) / np.std(ecg_lead)

# Load ECG data (example placeholder, replace with your actual data)
# Assume ecg_data is a NumPy array of shape (12, N) where N is the number of samples
# ecg_data = np.load('ecg_data.npy')

# Example synthetic ECG data for testing (12 leads, 5000 samples)
np.random.seed(42)
ecg_data = np.random.randn(12, 5000)

# Remove baseline drift from each lead
ecg_data_detrended = np.array([remove_baseline_drift(lead) for lead in ecg_data])

# Separate into Einthoven, Goldberger, and Wilson leads
einthoven_leads = ecg_data_detrended[0:3]  # Leads I, II, III
goldberger_leads = ecg_data_detrended[3:6] # Leads aVR, aVL, aVF
wilson_leads = ecg_data_detrended[6:12]    # Leads V1-V6

# Normalize each set of leads
einthoven_leads = np.array([normalize_lead(lead) for lead in einthoven_leads])
goldberger_leads = np.array([normalize_lead(lead) for lead in goldberger_leads])
wilson_leads = np.array([normalize_lead(lead) for lead in wilson_leads])

# Combine leads back into a single array
ecg_data_normalized = np.vstack((einthoven_leads, goldberger_leads, wilson_leads))

# The processed ECG data is now stored in ecg_data_normalized
print("ECG data preprocessing complete. Processed data shape:", ecg_data_normalized.shape)
