import os
import numpy as np
import torch
import pydicom as dicom
import matplotlib.pyplot as plt


### Read in the cine mri data from local folder
mri_path = os.chdir('/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/LGE')
cine_path = '/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/LGE'
files = os.listdir(mri_path)
for patients in files:
    if patients != ".DS_Store": 
        data = ("/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/LGE/" + patients)
        subfiles = sorted(os.listdir(data))
        volume = []
        for f in subfiles:
            dat = dicom.dcmread(data + "/" +  f)
            volume.append(dat.pixel_array)

for i in range(len(volume)):
    plt.imshow(volume[i], cmap = "gray")
    plt.savefig("/Users/nadjagruber/Documents/ECG_MRI_Project/data/volume"+str(i)+".png")

import nibabel as nib
from totalsegmentator.python_api import totalsegmentator



import nibabel as nib
from totalsegmentator.python_api import totalsegmentator



input_path = "/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/LGE/Zwicknagl^Hannes_20201030"
output_path = "/Users/nadjagruber/Documents/ECG_MRI_Project/data"

if __name__ == "__main__":
    # option 1: provide input and output as file paths
    totalsegmentator(input_path, output_path, task = "total_mr")
    
    # option 2: provide input and output as nifti image objects
    input_img = nib.load(input_path)
    output_img = totalsegmentator(input_img)
    nib.save(output_img, output_path)    
                