import os
import numpy as np
import torch
import pydicom as dicom
import matplotlib.pyplot as plt
import SimpleITK as sitk 


### Read in the cine mri data from local folder
mri_path = os.chdir('/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_preprocessing/Dataset_cine_example')
cine_path = '/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_preprocessing/Dataset_cine_example'
files = os.listdir(mri_path)
print(files)
img_dir = files[0]
print(img_dir)
dummy_img = sitk.GetArrayFromImage(sitk.ReadImage(img_dir))
print(dummy_img.shape)
for i in range(30):
    plt.imshow(dummy_img[i][5], cmap = "gray")
    plt.savefig("/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_preprocessing/output/volume"+str(i)+".png")