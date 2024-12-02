import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from airPLS import airPLS
import pandas as pd
from datetime import datetime, timedelta




results = []
''' this is the script for the preprocessing of the ECG data and extraction of label of the excel file'''
# Directory to save the tensor and label .pt files
tensor_save_dir = '/Users/nadjagruber/Documents/ECG_MRI_Project/ecg_preprocessing_codes/data/IMH_tensors/'
label_save_dir = '/Users/nadjagruber/Documents/ECG_MRI_Project/ecg_preprocessing_codes/data/IMH_tensors/'
os.makedirs(tensor_save_dir, exist_ok=True)
os.makedirs(label_save_dir, exist_ok=True)

def standard_normalize(ecg_signal):
    # Normalize each lead (if multiple leads exist)
    mean = np.mean(ecg_signal)
    std_dev = np.std(ecg_signal)
    
    normalized_signal = (ecg_signal - mean) / std_dev
    return normalized_signal

def pad_with_boundary_values(leads, pad_width=10):
    return np.pad(leads, ((0, 0), (pad_width, pad_width)), mode='edge')

# Read the Excel file
excel_file = '/Users/nadjagruber/Documents/ECG_MRI_Project/ecg_preprocessing_codes/marinastemi.xlsx'
df = pd.read_excel(excel_file)
#print(df['Laufnummer_Marina_STEMI'])
df['Revasc_time_(dd.mm.yyyy hh:mm)'] = pd.to_datetime(df['Revasc_time_(dd.mm.yyyy hh:mm)'], format='%d.%m.%y %H:%M')

# Loop through the files
i = 1
directory = '/Users/nadjagruber/Documents/ECG_MRI_Project/ECG_NPZ'
files = os.listdir(directory)
data = []
labels = []
patientIDs = []
for f in files:

    if f != '.DS_Store' and 'xlsx' not in f:
        i += 1
        # Find the corresponding label from the Excel file
      #  print(f"Laufnummer_Marina_STEMI column: {df['Laufnummer_Marina_STEMI']}")  # Print the series correctly
       # print(int(f.split('_')[-1]))
        label_row = df[df['Laufnummer_Marina_STEMI'] == int(f)] # Adjusted to match f with Laufnummer
       # print(label_row)
       # print(int(f) in df['Laufnummer_Marina_STEMI'])
        if not label_row.empty and (label_row['IMH_BL_Nein=0_Ja=1'].values[0]) in [0, 1]:
            print(f )

            label = label_row['IMH_BL_Nein=0_Ja=1'].values[0]
            revasc_date = label_row['Revasc_time_(dd.mm.yyyy hh:mm)'].values[0]
            # Convert the revasc_date to datetime
            revasc_datetime = pd.to_datetime(revasc_date)
            patientIDs.append(f)
        else:
            print(f"Warning: No label found for {f}. Skipping this file.")
            continue

        # One-hot encode the label
        one_hot_label = torch.zeros(2)
        one_hot_label[int(label)] = 1
        
            # Filter the .npy files in ecg_dates and find the closest to revasc date
        ecg_dates = os.listdir(os.path.join(directory, f))
        print(ecg_dates)
        ecg_dates = [e for e in ecg_dates if e.endswith('npz')]
        
        closest_file = None
        smallest_diff = float('inf')
        

        for ecg_file in ecg_dates:
            # Extract datetime from file name in the format "laufnummer_yyyymmdd_hhmmss.npy"
            file_datetime_str = '_'.join(ecg_file.split('_')[1:3]).split('.')[0]
            file_datetime = datetime.strptime(file_datetime_str, '%Y%m%d_%H%M%S')

            # Define the valid time window (30 minutes before to 6 hours after revasc_datetime)
            lower_bound = revasc_datetime - timedelta(hours=1)
            upper_bound = revasc_datetime + timedelta(hours=6)

            # Check if the file's datetime is within the valid window
            if lower_bound <= file_datetime <= upper_bound:
                time_diff = abs((file_datetime - revasc_datetime).total_seconds())

                # Update if this file is the closest so far
                if time_diff < smallest_diff:
                    smallest_diff = time_diff
                    closest_file = ecg_file

        if closest_file:
            print(f"Closest file: {closest_file}")
        else:
            print("No matching date found within the specified time window.")

        
        # Now load and process only the closest file
        if closest_file:
            results.append({
            'Name': f,
            'Closest File': closest_file,
            'Revasc DateTime': revasc_datetime, 
            'IMH label': label
            })
            labels.append(one_hot_label)



            path = os.path.join(directory, f)
            np_data = np.load(os.path.join(path, closest_file))['ecg']
            np_data = np.transpose(np_data)

            # Assume `np_data` is array of shape (12, 2500)
            einthoven_leads = np_data[0:3, :]     # Leads I, II, III
            goldberger_leads = np_data[3:6, :]    # Leads aVR, aVL, aVF
            wilson_leads = np_data[6:12, :]       # Leads V1 to V6

            # Pad each set of leads with 10 values on both sides
            einthoven_leads_padded = pad_with_boundary_values(einthoven_leads, pad_width=20)
            goldberger_leads_padded = pad_with_boundary_values(goldberger_leads, pad_width=20)
            wilson_leads_padded = pad_with_boundary_values(wilson_leads, pad_width=20)

            ## save baseline shifts, check if for all ecgs separatly!!!!!!
            e_s = np.apply_along_axis(airPLS, 1, einthoven_leads_padded)
            g_s = np.apply_along_axis(airPLS, 1, goldberger_leads_padded)
            w_s = np.apply_along_axis(airPLS, 1, wilson_leads_padded)

            # Apply baseline correction to each lead separately (after padding)
            einthoven_leads_corrected = einthoven_leads_padded - e_s
            goldberger_leads_corrected = goldberger_leads_padded - g_s
            wilson_leads_corrected = wilson_leads_padded - w_s


            # Remove padding after baseline correction to match the original shape
            einthoven_leads_corrected = einthoven_leads_corrected[:, 10:-10]
            goldberger_leads_corrected = goldberger_leads_corrected[:, 10:-10]
            wilson_leads_corrected = wilson_leads_corrected[:, 10:-10]

            # Normalize each group of leads separately
            einthoven_normalized = standard_normalize(einthoven_leads_corrected)
            goldberger_normalized = standard_normalize(goldberger_leads_corrected)
            wilson_normalized = standard_normalize(wilson_leads_corrected)

            # Stack arrays vertically to return to shape (12, 2500)
            processed_ecg_data = np.vstack((einthoven_leads_corrected, goldberger_leads_corrected, wilson_leads_corrected))
            processed_norm_ecg_data = np.vstack((einthoven_normalized, goldberger_normalized, wilson_normalized))
            shifts = np.vstack((e_s, g_s, w_s))
          #  print(processed_ecg_data.shape)
            data.append(np_data[:, :2500])
            
            # Plot and save as image
            plt.figure()
            plt.plot(np_data[2, :2500], label='original')
            plt.plot(processed_ecg_data[2, :2500], label='ALS-corrected')
            plt.plot(processed_norm_ecg_data[2, :2500], label='final')
            plt.plot(shifts[2, :2500], label='bl shift')

            plt.title('IMH ' + str(label))

            plt.legend()
            plt.savefig(os.path.join(tensor_save_dir, f"tensor_{i}.png"))
            plt.close()
            print('ecg ' + str(i) + ' saved in' + os.path.join(tensor_save_dir, f"tensor_{i}.png"))
            # Convert results to a DataFrame and write to Excel
            df1 = pd.DataFrame(results)
            df1.to_excel('/Users/nadjagruber/Documents/ECG_MRI_Project/Github/MMCL-ECG-CMR/ecg_preprocessing/summary_ecg_data_revasctime_labelpresent.xlsx', index=False)
data_np = np.asarray(data)
data_torch = torch.tensor(data_np)

lab = torch.stack(labels, 0)
print(lab.shape)
print(data_torch.shape)
data_torch = data_torch.to(torch.float32)
# Save the tensor as a .pt file
torch.save(lab[:50], os.path.join(label_save_dir + '/train', "labels.pt"))
torch.save(data_torch[:50], os.path.join(tensor_save_dir + '/train', "tensor.pt"))

torch.save(lab[50:], os.path.join(label_save_dir + '/val', "labels.pt"))
torch.save(data_torch[50:], os.path.join(tensor_save_dir + '/val', "tensor.pt"))
#print(data_torch.to(torch.float32).dtype)

print(i)
print(patientIDs)