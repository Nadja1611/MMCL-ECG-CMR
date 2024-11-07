import os
import numpy as np
import torch
import pydicom as dicom
import matplotlib.pyplot as plt


### Read in the cine mri data from local folder
mri_path = os.chdir('/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/BL/Cine')
cine_path = '/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/BL/Cine/'
output_dir = '/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_preprocessing/Tensors_cine/'
files = os.listdir(mri_path)
Vol_total = []

for patients in files:
    if patients != ".DS_Store": 
        data = (cine_path + patients)
        subfiles = os.listdir(data)
        ''' now we loop through the levels of the MRI '''
        for f in subfiles:
            volume = []
            patient_positions = []
            path = data + "/" +  f
            for cine in os.listdir(path):
                dat = dicom.dcmread(path + "/" +  cine)
                volume.append(dat.pixel_array)
                patient_positions.append(dat.ImagePositionPatient)
                # Convert volume and positions to PyTorch tensors
            volume_tensor = torch.tensor(volume)
            positions_tensor = torch.tensor(patient_positions)
        
            # Save the tensors to a separate file for each patient
            patient_data = {
            'volume': volume_tensor.to(torch.float32),
            'positions': positions_tensor.to(torch.float32)
            }

    
            # Define the file name for the current patient
            patient_file = output_dir + "/patients" +'_'+str(f) + "cine_data.pt"
            
            # Save the patient data to a Torch file
            torch.save(patient_data, patient_file)

            print(f"Data for {patients} saved to {patient_file}")
        Vol_total.append(volume)
            

print(len(Vol_total))
print(len(Vol_total[0]))
print(len(Vol_total[0][0]))
