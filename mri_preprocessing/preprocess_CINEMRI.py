
### Read in the cine mri data from local folder
mri_path = os.chdir('/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/Cine')
cine_path = '/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/Cine'
files = os.listdir(mri_path)
for patients in files:
    patient_volume = []
    patient_positions = []
    if patients != ".DS_Store": 
        data = ("/Users/nadjagruber/Documents/ECG_MRI_Project/data/ECG2MRI/BL/Cine/" + patients)
        subfiles = os.listdir(data)
        volume = []
        for patient in subfiles:
            dat = dicom.dcmread(data + "/" +  patient)
            print(dat)
            patient_volume.append(dat.pixel_array)
            patient_positions.append(dat.ImagePositionPatient)

for patient in os.listdir(data_dir):

    # Full path to the patient's folder
    patient_dir = os.path.join(data_dir, patient)
    print(patient_dir)
    if "Stanojevic^Miodrag_20170317"!=patient:
        ### this patient has six instead of 3 t2 star images, if i delete 3, the file is not readable anymore
        if os.path.isdir(patient_dir):

            # Loop through all DICOM images for the current patient
            for image_file in os.listdir(patient_dir):
                # Full path to the DICOM file
                image_path = os.path.join(patient_dir, image_file)
                
                # Read the DICOM file
                dat = dicom.dcmread(image_path)
                
                # Extract the pixel data (volume) and the image position
                patient_volume.append(dat.pixel_array)
                print(dat.pixel_array.shape)
                patient_positions.append(dat.ImagePositionPatient)
            
            # Convert volume and positions to PyTorch tensors
            volume_tensor = torch.tensor(patient_volume)
            positions_tensor = torch.tensor(patient_positions)
            
            # Save the tensors to a separate file for each patient
            patient_data = {
                'volume': volume_tensor.to(torch.float32),
                'positions': positions_tensor.to(torch.float32)
            }
        else:
            print("this is not a directoy")
        
        # Define the file name for the current patient
        patient_file = output_dir + f"{patient}_data.pt"
        
        # Save the patient data to a Torch file
        torch.save(patient_data, patient_file)

        print(f"Data for {patient} saved to {patient_file}")

print(patient_data)