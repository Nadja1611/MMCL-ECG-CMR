import os
import numpy as np
import torch
import pydicom as dcm
import torch.nn.functional as F
import pandas as pd 
import matplotlib.pyplot as plt

def pad_volume(volume, target_shape):
    padded_volume = []
    for slice_tensor in volume:
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

# Define the directory containing patient data
data_dir = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/T2_STAR/"

def generate_tensors_and_labels_t2star(data_dir, df):
    patient_volumes = []
    Pat = []
    labels = []
    laufnummer = []
    # Filter patients based on Excel file criteria and slice count
    valid_patient_rows = df[df['IMH_BL_Nein=0_Ja=1'].isin([0, 1])]  # Only include labels 0 and 1
    valid_patients = valid_patient_rows['Laufnummer_Marina_STEMI'].astype(str)
    converted_list = [str(int((float(item)))) for item in valid_patients]
    max_height, max_width = 0, 0

    # First pass to find the maximum height and width across all valid patients
    for patient in sorted(os.listdir(data_dir)):

        if (patient.split('_')[0])  in (converted_list):  # Check if patient is valid by name
            laufnummer.append(patient.split('_')[0])
            patient_dir = os.path.join(data_dir, patient)
            slice_files = [f for f in os.listdir(patient_dir) if f != '.DS_Store']

            label_row = df[df['Laufnummer_Marina_STEMI'] == int(patient.split('_')[0])] # Adjusted to match f with Laufnummer
            if not label_row.empty:
                label = label_row['IMH_BL_Nein=0_Ja=1'].values[0]
            # Convert the revasc_date to datetime
            else:
                print(f"Warning: No label found for {patient}. Skipping this file.")
                continue       
            if len(slice_files) != 3:
                print(f"Skipping {patient} due to insufficient slices (found {len(slice_files)})")
                continue
            elif len(slice_files) == 3 and label == 1 or label == 0:  # Only consider patients with exactly 3 slices
                print(len(slice_files))
                patient_volume = []
                for image_file in sorted(slice_files):
                    image_path = os.path.join(patient_dir, image_file)
                    dat = dcm.dcmread(image_path)
                    image = dat.pixel_array
                    image = image - np.mean(image)
                    image = image/np.std(image)
                    patient_volume.append(torch.tensor(image, dtype=torch.float32))
                            # One-hot encode the label
                max_height = max(max_height, patient_volume[0].shape[0])
                max_width = max(max_width, patient_volume[0].shape[1])            
                padded_volume = pad_volume(torch.stack(patient_volume), (max_height, max_width))
                Pat.append((padded_volume))
                one_hot_label = torch.zeros(2)
                one_hot_label[int(label)] = 1
                labels.append(one_hot_label)
            label = 2   
    return torch.stack(Pat), torch.stack(labels), laufnummer            

    

    


# Generate tensors and labels, ensuring they match the filtered criteria
patients_data, labels_tensor, laufnummer = generate_tensors_and_labels_t2star(data_dir, df)
#### check if images are correct
for i in range(3):
    plt.figure()
    plt.imshow(patients_data[55,i], cmap = 'inferno')
    plt.savefig('/Users/nadjagruber/Documents/ECG_MRI_Project/tensors'+str(i)+'.png')

print(labels_tensor)
print(torch.sum(labels_tensor[:,1]))
# Save to .pt files
torch.save(patients_data, '/Users/nadjagruber/Documents/ECG_MRI_Project/processed_t2star_data.pt')
torch.save(labels_tensor, '/Users/nadjagruber/Documents/ECG_MRI_Project/processed_t2star_labels.pt')

print(f"Saved stacked T2* data to 'processed_t2star_data.pt' with shape {len(patients_data)}")
print(f"Saved labels to 'processed_t2star_labels.pt' with {len(labels_tensor)} entries")
print(f"Saved data of the following patients  entries {laufnummer}")


