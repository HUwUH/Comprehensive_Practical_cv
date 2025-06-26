import numpy as np
import nibabel as nib
import os


def npy_to_nii(npy_path, nii_path, affine=None, header=None, dtype=None):
    """
    将.npy文件转换为.nii格式并保存到指定路径

    参数:
    npy_path: str - 输入的.npy文件路径
    nii_path: str - 输出的.nii文件路径
    affine: np.array, 可选 - 4x4仿射变换矩阵，默认为单位矩阵
    header: nib.Nifti1Header, 可选 - NIfTI头部信息
    dtype: data-type, 可选 - 输出数据类型

    返回:
    nib.Nifti1Image - 创建的NIfTI图像对象

    示例:
    npy_to_nii('input.npy', 'output.nii')
    """
    try:
        # 加载.npy文件
        npy_data = np.load(npy_path)

        # 检查数据维度
        if npy_data.ndim < 2 or npy_data.ndim > 4:
            raise ValueError(f"不支持的数据维度: {npy_data.ndim}。支持2D, 3D或4D数据")

        # 设置默认仿射矩阵（如果没有提供）
        if affine is None:
            affine = np.eye(4)  # 单位矩阵

            # 根据数据维度设置合理的体素大小
            voxel_size = [1.0] * min(npy_data.ndim, 3)
            if len(voxel_size) == 2:  # 2D数据
                affine[0, 0] = voxel_size[0]  # x方向体素大小
                affine[1, 1] = voxel_size[1]  # y方向体素大小
            elif len(voxel_size) >= 3:  # 3D或4D数据
                affine[0, 0] = voxel_size[0]  # x方向体素大小
                affine[1, 1] = voxel_size[1]  # y方向体素大小
                affine[2, 2] = voxel_size[2]  # z方向体素大小

        # 创建NIfTI图像对象
        if dtype is not None:
            npy_data = npy_data.astype(dtype)

        nii_img = nib.Nifti1Image(npy_data, affine, header=header)

        # 确保输出目录存在
        output_dir = os.path.dirname(nii_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存NIfTI文件
        nib.save(nii_img, nii_path)

        print(f"成功转换: {npy_path} -> {nii_path}")
        print(f"输出维度: {nii_img.shape}")
        print(f"数据类型: {nii_img.get_data_dtype()}")

        return nii_img

    except Exception as e:
        print(f"转换失败: {str(e)}")
        raise