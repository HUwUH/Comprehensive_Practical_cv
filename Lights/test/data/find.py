import os
import pydicom

dicom_folder = r"D:\MyFile\LIDC-IDRI\LIDC-IDRI-0017"
target_uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.305973183883758685859912046949"

found = False
for root, dirs, files in os.walk(dicom_folder):
    for file in files:
        if file.lower().endswith('.dcm'):
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                if ds.SOPInstanceUID == target_uid:
                    print(f"找到文件: {file}")
                    found = True
                    break
            except Exception:
                continue
    if found:
        break

if not found:
    print("未找到对应的DICOM文件。")