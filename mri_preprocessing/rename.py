# -*- encoding: utf-8 -*-

import pandas as pd
import os 
import re

' specify for which modality you want to rename the folders '
modality = 'T2_STAR'

file = '/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/Tabelle_Datensatz.xlsx'
df = pd.read_excel(file)
# Sample lists of names and numbers
surnames = df['Name']
names = df['Vorname']
numbers = df["Laufnummer"]
print(surnames)
# Function to combine surname and name in the format "surname^name_"
def combine_surnames_names(surnames, names):
    combined = []
    for surname, name in zip(surnames, names):
        combined.append(f"{surname}^{name}_")
    return combined
import re

def normalize_string(s):
    # Remove spaces, special characters, and umlauts (ä, ö, ü, etc.)
    return re.sub(r'[^\w]', '', s).replace('ä', 'a').replace('ö', 'o').replace('ü', 'u').replace('Ü', 'U').replace('Ä', 'A').replace('Ö', 'O')

def are_strings_equal(s1, s2):
    return normalize_string(s1) in normalize_string(s2)


# Get the combined names§§
combined_names = combine_surnames_names(surnames, names)
print(combined_names)
# Print the combined names
#print(combined_names)
# Sample folders (they could be actual folders in a directory or strings representing folder names)
folders = "/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/BL/" + modality




# Dictionary mapping names to numbers
name_to_number = dict(zip(combined_names, numbers))

print(name_to_number.items())


def replace_before_underscore(s, replacement):
    # Ersetze alles vor dem Unterstrich und den Unterstrich selbst durch 'replacement'
    return re.sub(r'^[^_]*_', replacement + '_', s)

# Function to replace names with numbers in folder names
def replace_name_with_number(folders, name_to_number, directory):
    updated_folders = []
    for name, number in name_to_number.items():
        
        newfolder = "".join(folder.split())
        #print(newfolder)

        if are_strings_equal(name,newfolder)==True:

                # Replace the name with the corresponding number

                updated_folder = replace_before_underscore(newfolder, str(number))

                updated_folders.append(updated_folder)

                # Construct full paths for renaming
                old_folder_path = os.path.join(directory, folder)
                new_folder_path = os.path.join(directory, updated_folder)
                
                # Rename the folder
                os.rename(old_folder_path, new_folder_path)
                print("Folders renamed successfully!")
                break  # Exit the loop once replacement is done

    return updated_folders
for folder in os.listdir(folders):
# Replace names with numbers
    new_folder_names = replace_name_with_number(folder, name_to_number, "/Users/nadjagruber/Documents/ECG_MRI_Project/ECG2MRI_new/BL/" + modality)
   # print(new_folder_names)    



