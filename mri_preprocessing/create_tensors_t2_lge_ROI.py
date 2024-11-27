import os
import numpy as np
import torch
import pydicom as dcm
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import resize
from operator import itemgetter
import torch.nn.functional as F



def constant_pad(x, size, c=0):
    padding_size = ((0, size - x.shape[0]), (0, size - x.shape[1]))
    return np.pad(x, padding_size, mode='constant', constant_values=c)


outputdir = '/Users/nadjagruber/Documents/ECG_MRI_Project/Data_T2STAR_CL'
data_dir = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/T2_STAR/"
data_dir_lge = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/BL/LGE/"
data_dir_lge_seg = "/Users/nadjagruber/Documents/ECG_MRI_Project/MRI_data_preprocessed/LGE_segmentations_LV_Scar_MVO"
excel_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/filtered_marinastemi_label_and_t2star_and_lge_available.xlsx'

def find_closest_indices(list_a, list_b):
    """
    Finds the indices in list_b that are closest to the values in list_a.
    
    Args:
        list_a (list of lists): A list containing sublists of target values.
        list_b (list of lists): A list containing sublists of reference values.
        
    Returns:
        list of lists: Indices of the closest values from list_b for each value in list_a.
    """
    closest_indices = []
    
    for sublist_a, sublist_b in zip(list_a, list_b):
        indices = []
        for value in sublist_a:
            # Find the index of the closest value in sublist_b
            closest_index = min(range(len(sublist_b)), key=lambda i: abs(sublist_b[i] - value))
            indices.append(closest_index)
        closest_indices.append(indices)
    
    return closest_indices





print('Starting the process...')

# Read Excel file and parse revascularization times
df = pd.read_excel(excel_file)
df['Revasc_time_(dd.mm.yyyy hh:mm)'] = pd.to_datetime(df['Revasc_time_(dd.mm.yyyy hh:mm)'], format='%d.%m.%y %H:%M')

def generate_tensors_and_labels_t2star(data_dir, data_dir_lge, data_dir_lge_seg, df):
    Pat, labels, Pat_lge, Pat_lge_seg, laufnummer, laufnummer_lge, laufnummer_lge_seg = [], [], [], [], [], [], []
    valid_patient_rows = df[df['IMH_BL_Nein=0_Ja=1'].isin([0, 1])]
    valid_patients = valid_patient_rows['Laufnummer_Marina_STEMI'].astype(str)
    converted_list = [str(int(float(item))) for item in valid_patients]
    slice_locations , slice_locations_lge, slice_locations_lge_seg = [], [], []
    list_of_slice_locations_lge, list_of_slice_locations_lge_seg, list_of_slice_locations_t2 = [], [], []
    for patient in sorted(os.listdir(data_dir)[:10]):
        slice_locations = []
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
            slice_data, slice_data_lge, slice_data_lge_seg = [], [], []
            for image_file in sorted(slice_files):
                image_path = os.path.join(patient_dir, image_file)
                dat = dcm.dcmread(image_path)
                slice_location = float(dat.SliceLocation)
                slice_locations.append(slice_location)
                print(dat.InstanceNumber)
                image = dat.pixel_array
                print('t2' + str(image.shape))
                image = (image - image.min()) / (image.max() - image.min())
                
                patient_volume.append(torch.tensor(image, dtype=torch.float32))
                                # Append the tensor and its slice location as a tuple
                slice_data.append((slice_location, torch.tensor(image, dtype=torch.float32)))

        # Sort the slice_data list by slice location
            #slice_data = sorted(slice_data, key=itemgetter(0))

        # Extract the reordered tensors
            #patient_volume = [item[1] for item in slice_data]
            padded_volume = torch.stack(patient_volume)
            print(f"Padded volume shape for {patient}: {padded_volume.shape}")
            Pat.append(padded_volume)
            list_of_slice_locations_t2.append(slice_locations)

    # Process LGE data
    j = -1
    for patient in sorted(os.listdir(data_dir_lge)[:10]):
        slice_locations_lge = []
        if patient != '.DS_Store':
            if (patient.split('_')[0]) not in converted_list:
                print(f"No LGE data for {patient.split('_')[0]}")
                continue
            elif (patient.split('_')[0]) in converted_list:
                laufnummer_lge.append(patient.split('_')[0])
                j += 1

                patient_dir_lge = os.path.join(data_dir_lge, patient)
                slice_files = [f for f in os.listdir(patient_dir_lge) if f != '.DS_Store']
                print(f"Processing LGE {patient}: Found {len(slice_files)} slices")
                slice_data, slice_data_lge, slice_data_lge_seg = [], [], []
                patient_volume_lge = []
                for image_file in sorted(slice_files):
                    image_path = os.path.join(patient_dir_lge, image_file)
                    dat = dcm.dcmread(image_path)
                    # Get the slice location
                    slice_location = float(dat.SliceLocation)
                    slice_locations_lge.append(slice_location)
                    
                    # Preprocess the image
                    image = dat.pixel_array
                    shape_lge = image.shape
                    print('lge' + str(image.shape))

                    ## zero padding as some images are rectangular
                    image = constant_pad(image, 256, c=0) 
        
                    image = (image - image.min()) / (image.max() - image.min())
                    patient_volume_lge.append(torch.tensor(image, dtype=torch.float32))

                    # Append the tensor and its slice location as a tuple
                    slice_data_lge.append((torch.tensor(image, dtype=torch.float32)))
                # Assuming Pat[j] is a PyTorch tensor
                Pat[j] = Pat[j].cpu().numpy()  # Convert to NumPy array (optional to add `.cpu()` if on GPU)
                Pat[j] = resize(Pat[j], (3, shape_lge[0], shape_lge[1]))  # Perform resizing
                K = np.zeros((3,256,256))
                for k in range(3):
                    K[k]=constant_pad(Pat[j][k], 256, 0)
                Pat[j] = torch.tensor(K, dtype=torch.float32)  # Convert back to PyTorch tensor                
    
                # Sort the slice_data list by slice location
                
                #slice_data_lge = sorted(slice_data_lge, key=itemgetter(0))

                # Extract the reordered tensors
                #patient_volume_lge = [item[1] for item in slice_data_lge]
                                
                padded_volume_lge = torch.stack(patient_volume_lge)
            #    padded_volume_lge = resize(np.array(padded_volume_lge), (10, int(padded_volume_lge.shape[1]), int(padded_volume_lge.shape[2])))
                print(padded_volume_lge.shape)
                Pat_lge.append(padded_volume_lge)
                list_of_slice_locations_lge.append(slice_locations_lge)
    # Process LGE data
    id = -1
    for patient in sorted(os.listdir(data_dir_lge_seg)[:10]):
        if patient != '.DS_Store':

            if (patient.split('_')[0]) not in converted_list:
                print(f"No LGE data for {patient.split('_')[0]}")
                continue
            elif (patient.split('_')[0]) in converted_list:
                id += 1
                laufnummer_lge_seg.append(patient.split('_')[0])

                patient_dir_lge_seg = os.path.join(data_dir_lge_seg, patient)
                slice_files = [f for f in os.listdir(patient_dir_lge_seg) if f != '.DS_Store']
                print(f"Processing LGE Seg {patient}: Found {len(slice_files)} slices")
                slice_data, slice_data_lge, slice_data_lge_seg = [], [], []
                slice_locations_lge_seg = []
                patient_volume_lge_seg = []
                for image_file in sorted(slice_files):
                    image_path = os.path.join(patient_dir_lge_seg, image_file)
                    dat = dcm.dcmread(image_path)
                    # Get the slice location
                    slice_location = float(dat.SliceLocation)
                    slice_locations_lge_seg.append(slice_location)
                    
                    # Preprocess the image
                    image = dat.pixel_array
                    ## zero padding as some images are rectangular
                    image = constant_pad(image, 256, c=0) 
        
                    image = (image - image.min()) / (image.max() - image.min())
                    patient_volume_lge_seg.append(torch.tensor(image, dtype=torch.float32))
                    # Append the tensor and its slice location as a tuple
                    slice_data_lge_seg.append((torch.tensor(image, dtype=torch.float32)))

                list_of_slice_locations_lge_seg.append(slice_locations_lge_seg)

                                
                padded_volume_lge_seg = torch.stack(patient_volume_lge_seg)
               # padded_volume_lge_seg = resize(np.array(padded_volume_lge_seg), (10, int(padded_volume_lge_seg.shape[1]), int(padded_volume_lge_seg.shape[2])))
                print(padded_volume_lge_seg.shape)
                Pat_lge_seg.append(padded_volume_lge_seg)
   

           

                        # Step 1: Find the closest slices for all patients
    # Assuming `indices_list` is generated from the `find_closest_indices` function
    indices_list = find_closest_indices(list_of_slice_locations_t2, list_of_slice_locations_lge)
    print(list_of_slice_locations_lge_seg)
    indices_list2 = find_closest_indices(list_of_slice_locations_t2, list_of_slice_locations_lge_seg)

    # Assuming Pat_lge and Pat_lge_seg are lists of tensors with different slice dimensions
    # Find the maximum size along the second dimension (number of slices)
    max_size = max(tensor.shape[1] for tensor in Pat_lge)

    # Pad tensors along the second dimension (number of slices) to make them all the same size
    Pat_lge_padded = [
        F.pad(tensor, (0, 0, 0, 0, 0, max_size - tensor.shape[0])) for tensor in Pat_lge
    ]

    Pat_lge_seg_padded = [
        F.pad(tensor, (0, 0, 0, 0, 0, max_size - tensor.shape[0])) for tensor in Pat_lge_seg
    ]

    # Verify that all tensors have the same shape after padding
    for idx, tensor in enumerate(Pat_lge_padded):
        print(f"Pat_lge_padded[{idx}] shape: {tensor.shape}")

    for idx, tensor in enumerate(Pat_lge_seg_padded):
        print(f"Pat_lge_seg_padded[{idx}] shape: {tensor.shape}")

    # Stack the padded tensors into a single tensor
    # Now all tensors should have the same slice size (max_size)
    Pat_lge = torch.stack(Pat_lge_padded)  # Shape: [8, max_size, 256, 256]
    Pat_lge_seg = torch.stack(Pat_lge_seg_padded)  # Shape: [8, max_size, 256, 256]
    print(indices_list)
    print(indices_list2)

    # Now process `indices_list` to select the required slices
    # indices_list: List[List[int]] with 8 sublists, each containing 3 indices
    Pat_lge_selected = torch.stack([
        torch.stack([Pat_lge[i, idx] for idx in indices_list[i]]) for i in range(len(indices_list))
    ])  # Shape: [8, 3, 256, 256]

    Pat_lge_seg_selected = torch.stack([
        torch.stack([Pat_lge_seg[i, idx] for idx in indices_list2[i]]) for i in range(len(indices_list2))
    ])  # Shape: [8, 3, 256, 256]

    # Assign the final tensors
    Pat_lge = Pat_lge_selected
    Pat_lge_seg = Pat_lge_seg_selected

    # Verify the final shapes
    print(f"Pat_lge shape: {Pat_lge.shape}")  # Expected: [8, 3, 256, 256]
    print(f"Pat_lge_seg shape: {Pat_lge_seg.shape}")  # Expected: [8, 3, 256, 256]
    for id in range(len(Pat)):
        plt.figure(figsize=(10,10))
        plt.subplot(3,3,1)
        plt.imshow(Pat[id][0])
        plt.subplot(3,3,2)
        plt.imshow(Pat[id][1])
        plt.subplot(3,3,3) 
        plt.imshow(Pat[id][2])
        plt.subplot(3,3,4)
        plt.imshow(Pat_lge[id][0], cmap = 'gray')
        plt.subplot(3,3,5)
        plt.imshow(Pat_lge[id][1], cmap = 'gray')
        plt.subplot(3,3,6) 
        plt.imshow(Pat_lge[id][2], cmap = 'gray')
        plt.title(str(labels[id]))
        plt.subplot(3,3,7) 
        plt.imshow(Pat_lge[id][0], cmap = 'gray')
        plt.imshow(Pat_lge_seg[id][0], alpha = 0.5)
        plt.subplot(3,3,8)
        plt.imshow(Pat_lge[id][1], cmap = 'gray')
        plt.imshow(Pat_lge_seg[id][1], alpha = 0.5)
        plt.subplot(3,3,9)
        plt.imshow(Pat_lge[id][2], cmap = 'gray')
        plt.imshow(Pat_lge_seg[id][2], alpha = 0.5)
        plt.savefig('/Users/nadjagruber/Documents/ECG_MRI_Project/Data_T2STAR_CL/'+ str(id) + '.png')
        plt.close()  
    return torch.stack(Pat), (((Pat_lge))), (((Pat_lge_seg))), torch.stack(labels), laufnummer, laufnummer_lge, laufnummer_lge_seg, slice_locations, slice_locations_lge, slice_locations_lge_seg

# Generate tensors and labels
patients_data, patients_data_lge, patients_data_lge_seg, labels_tensor, laufnummer, laufnummer_lge, laufnummer_lge_seg, slice_locations, slice_locations_lge, slice_locations_lge_seg = generate_tensors_and_labels_t2star(data_dir, data_dir_lge, data_dir_lge_seg, df)

# Save processed data
torch.save(patients_data, os.path.join(outputdir, 'processed_t2star_data.pt'))
torch.save(patients_data_lge, os.path.join(outputdir, 'processed_lge_data.pt'))
torch.save(patients_data_lge_seg, os.path.join(outputdir, 'processed_lge_seg.pt'))
torch.save(labels_tensor, os.path.join(outputdir, 'processed_t2star_labels.pt'))

print(f"Saved T2* data with shape {patients_data.shape}")
print(f"Saved LGE data with shape {(patients_data_lge.shape)}")
print(f"Saved labels with shape {labels_tensor.shape}")
print(f"Processed patients: {len(laufnummer)}")
print(f"Processed patients: {len(laufnummer_lge)}")
print(f"Processed patients: {len(laufnummer_lge_seg)}")
print(f"slice locations: {slice_locations}")
print(f"slice locations lge: {len(slice_locations_lge)}")
print(f"slice locations lge_seg: {len(slice_locations_lge_seg)}")




list1 = slice_locations_lge
list2 = slice_locations_lge_seg
print(list1)
print(list2)
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






