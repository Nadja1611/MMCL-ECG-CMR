import os
import numpy as np
import torch
import pydicom as dcm
import torch.nn.functional as F
import pandas as pd 


def pad_volume(volume, target_shape):
    """
    Pads the volume to the target shape.
    
    Args:
    - volume (torch.Tensor): Tensor of shape (num_slices, height, width).
    - target_shape (tuple): Desired shape (height, width).
    
    Returns:
    - Padded tensor of shape (num_slices, target_shape[0], target_shape[1]).
    """
    padded_volume = []
    for slice_tensor in volume:
        # Pad each slice to target dimensions
        padding = [
            0, target_shape[1] - slice_tensor.shape[1],  # Width padding
            0, target_shape[0] - slice_tensor.shape[0]   # Height padding
        ]
        padded_slice = F.pad(slice_tensor, padding, "constant", 0)
        padded_volume.append(padded_slice)
    
    return torch.stack(padded_volume)


# Read the marina stemi Excel file
excel_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/ecg_preprocessing_codes/marinastemi.xlsx'
df = pd.read_excel(excel_file)
df['Revasc_time_(dd.mm.yyyy hh:mm)'] = pd.to_datetime(df['Revasc_time_(dd.mm.yyyy hh:mm)'], format='%d.%m.%y %H:%M')

# Define the directories containing patient data
data_dir_t2star = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/T2_STAR/"
data_dir_lge = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/LGE/"


def generate_tensors_and_labels_t2star(data_dir):
    error = 0
    patient_volumes = []
    labels = []
    patient_names = []  # Store patient names for filtering later
    max_height, max_width = 0, 0
    
    # First pass to find the maximum height and width across all patient images
    for patient in sorted(os.listdir(data_dir)):
        patient_volume = []
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            patient_slices = [f for f in os.listdir(patient_dir) if f != '.DS_Store']
            if len(patient_slices) >= 3:  # Only proceed if the patient has at least 3 slices
                for image_file in sorted(patient_slices):
                    image_path = os.path.join(patient_dir, image_file)
                    dat = dcm.dcmread(image_path)
                    patient_volume.append(torch.tensor(dat.pixel_array, dtype=torch.float32))
                # Find max dimensions
                max_height = max(max_height, patient_volume[0].shape[0])
                max_width = max(max_width, patient_volume[0].shape[1])
                patient_names.append(patient)  # Save the patient name for later filtering
    
    # Second pass to pad images to the maximum dimensions and stack into tensor
    for patient in sorted(os.listdir(data_dir)):
        patient_volume = []
        patient_dir = os.path.join(data_dir, patient)
        if os.path.isdir(patient_dir):
            patient_slices = [f for f in os.listdir(patient_dir) if f != '.DS_Store']
            if len(patient_slices) >= 3:  # Only process if at least 3 slices exist
                for image_file in sorted(patient_slices):
                    image_path = os.path.join(patient_dir, image_file)
                    dat = dcm.dcmread(image_path)
                    patient_volume.append(torch.tensor(dat.pixel_array, dtype=torch.float32))
            
                # Pad each patient's volume to the maximum height and width
                padded_volume = pad_volume(torch.stack(patient_volume), (max_height, max_width))
                patient_volumes.append(padded_volume)
            else:
                error += 1  # Track if there is an error in the number of slices
    
    if patient_volumes:
        patients_data = torch.stack(patient_volumes)
    else:
        patients_data = torch.empty(0)  # In case no valid data is found

    return patients_data, patient_names, error


def extract_labels_and_filter(data_dir_t2star, data_dir_lge, df):
    labels = []
    patient_names = []
    filtered_rows = []  # To store the rows that meet the conditions
    
    for patient in sorted(os.listdir(data_dir_t2star)):  
        if patient != '.DS_Store' and 'xlsx' not in patient:
            patient_dir_lge = os.path.join(data_dir_lge, patient)
            if not os.path.isdir(patient_dir_lge) :  # Check if LGE data exists
                print('no lge found for ' + patient )
            
            if  os.path.isdir(patient_dir_lge) and len(os.listdir(patient_dir_lge)) > 0: 
                patient_names.append(patient)  # Keep track of patient name for later synchronization
                label_row = df[df['Laufnummer_Marina_STEMI'] == int(patient.split('_')[0])]  
                if not label_row.empty:
                    label = label_row['IMH_BL_Nein=0_Ja=1'].values[0]
                    if label == 1 or label == 0:
                        labels.append(label)
                        filtered_rows.append(label_row)
    
    # Concatenate all valid rows into a new DataFrame
    filtered_df = pd.concat(filtered_rows)
    return filtered_df, labels


# Generate tensors and labels, ensuring the labels are returned in the same order as volumes
patients_data, patient_names, error_count = generate_tensors_and_labels_t2star(data_dir_t2star)

# Extract the labels and filter the DataFrame based on the conditions, including the presence of LGE data
filtered_df, labels = extract_labels_and_filter(data_dir_t2star, data_dir_lge, df)

# Filter out rows where the number of slices is not 3 (or any other condition)
valid_patient_names = [patient for patient, volume in zip(patient_names, patients_data) if len(volume) == 3]

# Filter the DataFrame to keep only patients with 3 slices
filtered_df = filtered_df[filtered_df['Laufnummer_Marina_STEMI'].isin([int(patient.split('_')[0]) for patient in valid_patient_names])]

# Save the filtered DataFrame to a new Excel file
filtered_excel_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/filtered_marinastemi_label_and_t2star_and_lge_available.xlsx'
filtered_df.to_excel(filtered_excel_file, index=False)

print(f"Filtered Excel file saved to {filtered_excel_file}")
print(f"Number of patients processed: {patients_data.shape[0]}")
print(f"Number of patients used: {len(filtered_df)}")
print(filtered_df)
print(f"Number of errors (patients with < 3 slices): {error_count}")
