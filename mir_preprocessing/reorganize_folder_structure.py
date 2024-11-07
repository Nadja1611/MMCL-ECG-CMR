import os
import pydicom as dcm
import time

def reorder_levels_folders(base_folder):
    """
    Reorder the Level_X folders based on the SliceLocation of the slices and rename them accordingly.
    
    Args:
    - base_folder (str): Path to the base folder containing the Level_X subfolders.
    """
    levels = sorted(os.listdir(base_folder))
    all_level_positions = []

    # Loop through each Level_X folder and get the z-position
    for level in levels:
        if level != '.DS_Store':
            level_folder = os.path.join(base_folder, level)
            for file in os.listdir(level_folder)[:1]:
                if file != '.DS_Store':
                    dat = dcm.dcmread(os.path.join(level_folder, file))
                    slice_pos = dat.SliceLocation
                    all_level_positions.append((slice_pos, level_folder))

    # Sort the Level_X folders by SliceLocation (highest to lowest)
    all_level_positions = sorted(all_level_positions, key=lambda x: x[0], reverse=True)

    # Temporarily rename the folders to avoid conflicts
    for idx, (_, old_level_folder) in enumerate(all_level_positions):
        temp_name = os.path.join(base_folder, f"temp_{idx}_{time.time()}")
        os.rename(old_level_folder, temp_name)
        all_level_positions[idx] = (all_level_positions[idx][0], temp_name)

    # Rename the folders to their final names
    for new_index, (_, temp_name) in enumerate(all_level_positions):
        final_level_name = os.path.join(base_folder, f"Level_{new_index}")
        os.rename(temp_name, final_level_name)

def rename_files_by_instance_number(base_folder):
    """
    Orders and renames DICOM files within each Level_X folder by the InstanceNumber tag.
    
    Args:
    - base_folder (str): Path to the base folder containing the Level_X subfolders.
    """
    # Loop through each Level_X folder
    for level in sorted(os.listdir(base_folder)):
        level_folder = os.path.join(base_folder, level)
        
        if os.path.isdir(level_folder):
            dicom_files = [f for f in os.listdir(level_folder) ]
            files_with_instance = []

            # Read InstanceNumber for each DICOM file
            for file in dicom_files:
                file_path = os.path.join(level_folder, file)
                dat = dcm.dcmread(file_path)
                instance_number = dat.InstanceNumber
                files_with_instance.append((instance_number, file_path))
            
            # Sort files by InstanceNumber
            files_with_instance.sort(key=lambda x: x[0])

            # Rename files to their InstanceNumber
            for instance_number, file_path in files_with_instance:
                new_file_name = os.path.join(level_folder, f"{instance_number}.dcm")
                os.rename(file_path, new_file_name)
                print(new_file_name)
# Input folder path containing the Level_X folders
cine_dir = '/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/BL/Cine/'

# Process each patient folder
for patients in os.listdir(cine_dir):
    if patients != '.DS_Store':  # Ignore system files
        print(f"Processing patient: {patients}")
        base_folder = os.path.join(cine_dir, patients)
        reorder_levels_folders(base_folder)
        rename_files_by_instance_number(base_folder)
