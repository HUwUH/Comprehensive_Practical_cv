import pydicom
import os

# 指定DICOM文件所在的目录
dicom_dir = r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000'

# 遍历目录中的所有DICOM文件
for root, dirs, files in os.walk(dicom_dir):
    for file in files:
        if file.endswith('.dcm'):
            # 构建完整的文件路径
            file_path = os.path.join(root, file)
            # 读取DICOM文件
            dicom_file = pydicom.dcmread(file_path)
            # 打印DICOM文件的基本信息
            print(dicom_file)
            # 访问特定的DICOM标签
            patient_name = dicom_file.PatientName
            print(f"患者姓名: {patient_name}")