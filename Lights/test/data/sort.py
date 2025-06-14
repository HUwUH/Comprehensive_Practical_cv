import os
import pydicom

root_dir = r'D:\MyFile\LIDC-IDRI'

for lidc_idri_folder in os.listdir(root_dir):
    print(f"Processing LIDC-IDRI Folder: {lidc_idri_folder}")
    if not lidc_idri_folder.startswith("LIDC-IDRI-"):
        print(f"Skipping {lidc_idri_folder}, not a valid LIDC-IDRI folder.")
        continue
    lidc_idri_path = os.path.join(root_dir, lidc_idri_folder)
    if not os.path.isdir(lidc_idri_path):
        print(f"Skipping {lidc_idri_path}, not a directory.")
        continue

    # 收集所有 DICOM 文件及其 InstanceNumber
    dicom_files = []
    for file in os.listdir(lidc_idri_path):
        if file.endswith('.dcm'):
            file_path = os.path.join(lidc_idri_path, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                instance_number = getattr(ds, 'InstanceNumber', None)
                if instance_number is not None:
                    dicom_files.append((file_path, instance_number))
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

    # 按 InstanceNumber 排序
    dicom_files.sort(key=lambda x: x[1])

    # 先重命名为临时文件，避免重名覆盖
    temp_files = []
    for idx, (file_path, _) in enumerate(dicom_files):
        temp_path = file_path + '.tmp'
        os.rename(file_path, temp_path)
        temp_files.append(temp_path)

    # 再按顺序重命名为 000000.dcm、000001.dcm ...
    for idx, temp_path in enumerate(temp_files):
        new_name = f"{idx:06d}.dcm"
        new_path = os.path.join(lidc_idri_path, new_name)
        os.rename(temp_path, new_path)

    print(f"{lidc_idri_folder} 重命名完成。")