import pydicom

# 读取DICOM文件
dicom_file = pydicom.dcmread(r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0365\000000.dcm')

# 打印DICOM文件的基本信息
# print(dicom_file)

# 访问特定的DICOM标签
Manufacturer = dicom_file.Manufacturer
print(f"Manufacturer: {Manufacturer}")

SOP_Instance_UID = dicom_file.SOPInstanceUID
print(f"SOPInstanceUID: {SOP_Instance_UID}")

InstanceNumber = dicom_file.InstanceNumber
print(f"InstanceNumber: {InstanceNumber}")

ImagePosition = dicom_file.ImagePositionPatient
print(f"ImagePositionPatient: {ImagePosition}")

SliceThickness = dicom_file.SliceThickness
print(f"SliceThickness: {SliceThickness}")

BitsAllocation = dicom_file.BitsAllocated
print(f"BitsAllocation: {BitsAllocation}")

BitsStored = dicom_file.BitsStored
print(f"BitsStored: {BitsStored}")





