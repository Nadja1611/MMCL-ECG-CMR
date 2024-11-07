import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
import dicom2nifti



# The NIfTI file will be saved in the specified output directory
Levels = []
Images = []
index = 0
patients = sorted(os.listdir('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine'))[:2]
for pat in patients:
    print(pat)
    if pat != '.DS_Store':
        for levels in sorted(os.listdir('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine/' + pat)):
            if levels != '.DS_Store':
                levellist = sorted(os.listdir('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine/' + pat + "/" + levels))
                dicom_directory = ('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine/' + pat + "/" + levels + '/')
                output_file = '/Users/nadjagruber/Downloads/output/N.nii.gz'
                #print(levellist)
                for names in levellist[:1]:
                    print(names)
                    img = dcm.dcmread('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine/' + pat + "/" + levels + '/' + names)
                    dicom_directory = '/Users/nadjagruber/Documents/ECG_MRI_Project/Data_MRI/ECG2MRI/BL/Cine/' + pat + "/" + levels + '/' + names
                    print(dicom_directory)
                #    dicom2nifti.convert_dicom.dicom_array_to_nifti('/Users/nadjagruber/Downloads/output/', os.path.dirname(output_file))

                    img = img.pixel_array
                    Images.append(img)
