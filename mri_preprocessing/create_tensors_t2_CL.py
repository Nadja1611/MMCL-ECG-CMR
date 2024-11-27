import os
import numpy as np
import torch
import pydicom as dcm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from operator import itemgetter

outputdir = '/Users/nadjagruber/Documents/ECG_MRI_Project/Data_T2STAR_CL'
data_dir = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/T2_STAR/"
data_dir_lge = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/LGE/"
excel_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/filtered_marinastemi_label_and_t2star_and_lge_available.xlsx'

print('Starting the process...')

# Read Excel file and parse revascularization times
df = pd.read_excel(excel_file)
df['Revasc_time_(dd.mm.yyyy hh:mm)'] = pd.to_datetime(df['Revasc_time_(dd.mm.yyyy hh:mm)'], format='%d.%m.%y %H:%M')

def generate_tensors_and_labels_t2star(data_dir, data_dir_lge, df):
    Pat, labels, Pat_lge, laufnummer, laufnummer_lge = [], [], [], [], []
    valid_patient_rows = df[df['IMH_BL_Nein=0_Ja=1'].isin([0, 1])]
    valid_patients = valid_patient_rows['Laufnummer_Marina_STEMI'].astype(str)
    converted_list = [str(int(float(item))) for item in valid_patients]

    for patient in sorted(os.listdir(data_dir)[:1]):
        if (patient.split('_')[0]) in converted_list:
            laufnummer.append(patient.split('_')[0])
            patient_dir = os.path.join(data_dir, patient)
            slice_files = [f for f in os.listdir(patient_dir) if f != '.DS_Store']
            
            if len(slice_files) < 2:  # Skip patients with insufficient slices
                print(f"Skipping {patient} due to insufficient slices")
                continue
            
            label_row = df[df['Laufnummer_Marina_STEMI'] == int(patient.split('_')[0])]
            if label_row.empty:
                print(f"Warning: No label found for {patient}. Skipping.")
                continue
            label = label_row['IMH_BL_Nein=0_Ja=1'].values[0]
            # One-hot encode the label
            one_hot_label = torch.zeros(2)
            one_hot_label[int(label)] = 1
            labels.append(one_hot_label)
            patient_volume = []
            slice_data = []
            slice_locations , slice_locations_lge = [], []
            for image_file in sorted(slice_files):
                image_path = os.path.join(patient_dir, image_file)
                dat = dcm.dcmread(image_path)
                slice_location = float(dat.SliceLocation)
                slice_locations.append(slice_location)
                print(dat.InstanceNumber)
                image = dat.pixel_array
                mini = min(image.shape)
                diffi_vert = (image.shape[0] - mini) // 2
                diffi_hor = (image.shape[1] - mini) // 2

                # Crop to square
                image = image[diffi_vert:image.shape[0]-diffi_vert, diffi_hor:image.shape[1]-diffi_hor]
                # Resize and normalize
                image = resize(image, (256, 256))
                image = image[50:-50, 30:-70]  # Additional cropping
                image = (image - image.min()) / (image.max() - image.min())
                
                patient_volume.append(torch.tensor(image, dtype=torch.float32))
                                # Append the tensor and its slice location as a tuple
                slice_data.append((slice_location, torch.tensor(image, dtype=torch.float32)))

        # Sort the slice_data list by slice location
            slice_data = sorted(slice_data, key=itemgetter(0))

        # Extract the reordered tensors
            patient_volume = [item[1] for item in slice_data]
            padded_volume = torch.stack(patient_volume)
            print(f"Padded volume shape for {patient}: {padded_volume.shape}")
            Pat.append(padded_volume)

    # Process LGE data
    i=0
    for patient in sorted(os.listdir(data_dir_lge)[:1]):
        if patient != '.DS_Store':

            if (patient.split('_')[0]) not in converted_list:
                print(f"No LGE data for {patient.split('_')[0]}")
                continue
            elif (patient.split('_')[0]) in converted_list:
                laufnummer_lge.append(patient.split('_')[0])

                patient_dir_lge = os.path.join(data_dir_lge, patient)
                slice_files = [f for f in os.listdir(patient_dir_lge) if f != '.DS_Store']
                print(f"Processing LGE {patient}: Found {len(slice_files)} slices")

                patient_volume_lge = []
                for image_file in sorted(slice_files):
                    image_path = os.path.join(patient_dir_lge, image_file)
                    dat = dcm.dcmread(image_path)
                    # Get the slice location
                    slice_location = float(dat.SliceLocation)
                    slice_locations_lge.append(slice_location)
                    
                    # Preprocess the image
                    image = dat.pixel_array
                    mini = min(image.shape)
                    diffi_vert = (image.shape[0] - mini) // 2
                    diffi_hor = (image.shape[1] - mini) // 2

                    # Crop to square
                    image = image[diffi_vert:image.shape[0]-diffi_vert, diffi_hor:image.shape[1]-diffi_hor]
                    # Resize and normalize
                    image = resize(image, (256, 256))
                    image = image[50:-50, 30:-70]
                    image = (image - image.min()) / (image.max() - image.min())
                    
                    # Append the tensor and its slice location as a tuple
                    slice_data.append((slice_location, torch.tensor(image, dtype=torch.float32)))

                # Sort the slice_data list by slice location
                slice_data = sorted(slice_data, key=itemgetter(0))

                # Extract the reordered tensors
                patient_volume_lge = [item[1] for item in slice_data]
                                
                padded_volume_lge = torch.stack(patient_volume_lge)
                padded_volume_lge = resize(np.array(padded_volume_lge), (10, int(padded_volume_lge.shape[1]), int(padded_volume_lge.shape[2])))
                print(padded_volume_lge.shape)
                Pat_lge.append(padded_volume_lge)
                if laufnummer[i]!=laufnummer_lge[i]:
                    print('something is wrong')
                plt.figure(figsize=(10,8))
                plt.subplot(2,3,1)
                plt.imshow(padded_volume_lge[2])
                plt.subplot(2,3,2)
                plt.imshow(padded_volume_lge[4])
                plt.title(laufnummer_lge[i])
                plt.subplot(2,3,3)
                plt.imshow(padded_volume_lge[8])
                plt.subplot(2,3,4)
                plt.imshow(Pat[i][0])
                plt.subplot(2,3,5)
                plt.title(laufnummer[i])
                plt.imshow(Pat[i][1])
                plt.subplot(2,3,6)
                plt.imshow(Pat[i][2])
                plt.title(str(labels[i]))
                plt.savefig('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_T2STAR_CL/'+ patient + '.png')
                plt.close()                

                i += 1
                Pat_lge = [torch.tensor(array, dtype=torch.float32) for array in Pat_lge]
    return torch.stack(Pat), torch.stack(((Pat_lge))), torch.stack(labels), laufnummer, laufnummer_lge, slice_locations, slice_locations_lge

# Generate tensors and labels
patients_data, patients_data_lge, labels_tensor, laufnummer, laufnummer_lge, slice_locations, slice_locations_lge = generate_tensors_and_labels_t2star(data_dir, data_dir_lge, df)

# Save processed data
torch.save(patients_data, os.path.join(outputdir, 'processed_t2star_data.pt'))
torch.save(patients_data_lge, os.path.join(outputdir, 'processed_lge_data.pt'))
torch.save(labels_tensor, os.path.join(outputdir, 'processed_t2star_labels.pt'))

print(f"Saved T2* data with shape {patients_data.shape}")
print(f"Saved LGE data with shape {len(patients_data_lge)}")
print(f"Saved labels with shape {labels_tensor.shape}")
print(f"Processed patients: {len(laufnummer)}")
print(f"Processed patients: {len(laufnummer_lge)}")
print(f"slice locations: {slice_locations}")
print(f"slice locations lge: {slice_locations_lge}")




list1 = laufnummer
list2 = laufnummer_lge

# Find elements in list1 but not in list2
difference = list(set(list1) - set(list2))
print("In list1 but not in list2:", difference)

# Find elements in list2 but not in list1
difference = list(set(list2) - set(list1))
print("In list2 but not in list1:", difference)
print(len(list1) != len(set(list1)))
print(len(list2) != len(set(list2)))


# Find duplicates
seen = set()
duplicates = set()
for item in list2:
    if item in seen:
        duplicates.add(item)
    seen.add(item)

print("Duplicated elements:", list(duplicates))






