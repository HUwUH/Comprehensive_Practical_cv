import pydicom
import numpy as np
import cv2

def dcm_to_jpg(dcm_path, jpg_path, window_center=-600, window_width=1600):
    # 读取 DICOM 文件
    dcm = pydicom.dcmread(dcm_path)
    # 获取像素数据
    pixel_data = dcm.pixel_array.astype(np.float32)
    # 获取 DICOM 文件中的斜率和截距
    slope = dcm.RescaleSlope
    intercept = dcm.RescaleIntercept
    # 应用斜率和截距进行线性变换
    hu = pixel_data * slope + intercept

    # 计算窗位和窗宽的上下限
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # 进行窗宽窗位的调整
    windowed = np.clip(hu, window_min, window_max)
    # 归一化到 0-255 范围
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)

    # 保存为 JPEG 图片
    cv2.imwrite(jpg_path, windowed)

# 使用示例
dcm_path = r'D:\MyFile\LIDC-IDRI\LIDC-IDRI-0616\000043.dcm'
jpg_path = 'output_image.jpg'
dcm_to_jpg(dcm_path, jpg_path)