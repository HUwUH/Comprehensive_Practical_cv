import pydicom
import os

# 指定DICOM文件所在的目录
dicom_dir = r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0365'

# 遍历目录中的所有DICOM文件
for root, dirs, files in os.walk(dicom_dir):
    for file in files:
        # print("Processing file:", file)
        if file.endswith('.dcm'):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 读取DICOM文件
            dicom_file = pydicom.dcmread(file_path)
            # 打印DICOM文件的基本信息
            # print(dicom_file)
            # 访问特定的DICOM标签
            SOP_Instance_UID = dicom_file.SOPInstanceUID
            # print(f"SOPInstanceUID: {SOP_Instance_UID}")

            InstanceNumber = dicom_file.InstanceNumber
            print(f"InstanceNumber: {InstanceNumber}")

            ImagePosition = dicom_file.ImagePositionPatient
            print(f"ImagePositionPatient: {ImagePosition}")

            SliceLocation = dicom_file.SliceLocation
            print(f"SliceLocation: {SliceLocation}")

