# -*- encoding: utf-8 -*-
import os
import pydicom as dcm

# Specify for which modality you want to rename the folders
modality = 'LGE'
folders = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/" + modality

# Iterate over patient folders
for patient_folder in os.listdir(folders):
    if patient_folder != '.DS_Store':  # Exclude system files
        path = os.path.join(folders, patient_folder)
        print(f"Processing folder: {path}")
        
        dicom_files = []
        for dicom_file in os.listdir(path):
            dicom_path = os.path.join(path, dicom_file)
            
            try:
                dat = dcm.dcmread(dicom_path)  # Read DICOM file
                instance = dat.InstanceNumber  # Get Instance Number
                dicom_files.append((instance, dicom_file))  # Store instance and filename
            except Exception as e:
                print(f"Error reading file {dicom_file}: {e}")

        # Sort files by Instance Number
        dicom_files.sort(key=lambda x: x[0])

        # Rename files based on Instance Number
        for instance_number, old_filename in dicom_files:
            old_filepath = os.path.join(path, old_filename)
            new_filename = f"{instance_number:04d}.dcm"  # Zero-padded instance number
            new_filepath = os.path.join(path, new_filename)
            
            try:
                os.rename(old_filepath, new_filepath)
                print(f"Renamed {old_filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {old_filename}: {e}")
