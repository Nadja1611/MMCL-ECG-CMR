import dicom2nifti

import dicom2nifti
import os

# Path to the folder containing DICOM files
dicom_folder = '/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/LGE'

# Output path for the NIfTI file
output_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/LGE_nifti/output_file.nii.gz'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Convert DICOM to NIfTI
try:
    dicom2nifti.convert_directory(dicom_folder, os.path.dirname(output_file), compression=True, reorient=True)
    print(f"NIfTI file saved at: {output_file}")
except Exception as e:
    print(f"An error occurred: {e}")
