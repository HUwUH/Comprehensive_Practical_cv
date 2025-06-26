import os

def check_directory_structure(directory):
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            subfolders = [subfolder for subfolder in os.listdir(folder_path) 
                          if os.path.isdir(os.path.join(folder_path, subfolder))]
            
            # Check if the folder has no subfolders or more than 3 subfolders
            if len(subfolders) == 0 or len(subfolders) >= 3:
                print(f"Folder with invalid structure: {folder_path}")
                continue
            
            for subfolder_name in subfolders:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                sub_subfolders = [sub_subfolder for sub_subfolder in os.listdir(subfolder_path) 
                                  if os.path.isdir(os.path.join(subfolder_path, sub_subfolder))]
                
                # Check if the subfolder has no subfolders or more than 2 subfolders
                if len(sub_subfolders) == 0 or len(sub_subfolders) >= 2:
                    print(f"Subfolder with invalid structure: {subfolder_path}")
                elif len(sub_subfolders) == 1:
                    # print("ok")
                    continue

# Specify the directory to check
directory_a = "E:\work_files\praticalTraining_cv\LIDC-IDRI"
check_directory_structure(directory_a)