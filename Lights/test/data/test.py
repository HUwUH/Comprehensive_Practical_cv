import os
import pydicom
from collections import defaultdict
import logging
# 配置日志记录
logging.basicConfig(filename='lidc_idri_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

root_dir = r'D:\MyFile\LIDC-IDRI'
manufacturer_thickness = defaultdict(set)

for lidc_idri_folder in os.listdir(root_dir):
    logging.info(f"Processing {lidc_idri_folder}")
    print(f"Processing {lidc_idri_folder}")

    if not lidc_idri_folder.startswith("LIDC-IDRI-"):
        logging.warning(f"Skipping {lidc_idri_folder}, not a valid LIDC-IDRI folder.")
        print(f"Skipping {lidc_idri_folder}, not a valid LIDC-IDRI folder.")
        continue
    lidc_idri_path = os.path.join(root_dir, lidc_idri_folder)
    if not os.path.isdir(lidc_idri_path):
        logging.warning(f"Skipping {lidc_idri_path}, not a directory.")
        print(f"Skipping {lidc_idri_path}, not a directory.")
        continue

    for root, dirs, files in os.walk(lidc_idri_path):
        for file in files:
            if file.endswith('.dcm'):
                file_path = os.path.join(root, file)
                try:
                    dicom_file = pydicom.dcmread(file_path, stop_before_pixels=True)
                    manufacturer = getattr(dicom_file, 'Manufacturer', 'Unknown')
                    thickness = getattr(dicom_file, 'SliceThickness', None)
                    if thickness is not None:
                        manufacturer_thickness[manufacturer].add(thickness)
                except Exception as e:
                    print(f"读取文件 {file_path} 时出错: {e}")
                    logging.error(f"Error reading file {file_path}: {e}")
        break

    logging.info(f"Finished processing {lidc_idri_folder}")
    print(f"Finished processing {lidc_idri_folder}")
# 打印统计结果
for manufacturer, thicknesses in manufacturer_thickness.items():
    print(f"Manufacturer: {manufacturer}")
    print(f"SliceThickness: {sorted(thicknesses)}")