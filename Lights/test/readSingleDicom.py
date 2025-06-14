import pydicom

# 读取DICOM文件
dicom_file = pydicom.dcmread(r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000000.dcm')

# 打印DICOM文件的基本信息
print(dicom_file)

# 访问特定的DICOM标签
patient_name = dicom_file.PatientName
print(f"患者姓名: {patient_name}")
