import numpy as np
import nibabel as nib
import os
input_base_folder = ''#替换为输入文件地址
output_folder = '  '  # 替换为输出文件夹路径
# 获取所有的子文件夹
subfolders = [folder for folder in os.listdir(input_base_folder) if
              os.path.isdir(os.path.join(input_base_folder, folder))]
for subfolder in subfolders:
    # 构建 data.npy 的完整路径
    npy_file_path = os.path.join(input_base_folder, subfolder, 'data.npy')
    # print(npy_file_path)
    if os.path.exists(npy_file_path):
        print(f"Converting {npy_file_path} to NIfTI...")
        # 从NPY文件加载数据
        npy_data = np.load(npy_file_path)
        # 创建NIfTI对象
        nifti_img = nib.Nifti1Image(npy_data, affine=np.eye(4))  # 使用单位仿射矩阵
        # 构建输出路径
        output_nii_path = os.path.join(output_folder, f'{subfolder}.nii')
        # 保存为NII文件
        nib.save(nifti_img, output_nii_path)
        print(f"Saved {output_nii_path}")
    else:
        print(f"Skipped {subfolder}, data.npy not found.")