from utils.read_file import get_dcm_numpy_with_z
import os
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from typing import Tuple,List
from scipy.ndimage import zoom

def get_dcm_numpy(path: str) -> Tuple[str,List[dict]]:
    """
    输入文件路径或文件夹路径，返回 numpy 数组保存路径。
    如果是文件夹，会按 z 坐标排序并堆叠为 3D 数组。
    此外，返回一个list，保存全部层的{"z":z,"PixelSpaceYX":ds.PixelSpacing}。如果是单层，则没有z属性。
    """
    if not os.path.exists(path):
        return "path not exists"

    try:
        if os.path.isdir(path):
            # 收集所有 .dcm 文件路径
            paths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".dcm")]
            if not paths:
                return "no dcm files found"

            # 读取所有 DICOM 并排序
            imgs_with_info = get_dcm_numpy_with_z(paths)
            imgs_with_info.sort(key=lambda x: x[1]["z"])

            # 提取排序后的图像
            img_stack = np.stack([item[0] for item in imgs_with_info], axis=0)  # shape: [z, y, x]

            # 保存为 numpy 文件
            os.makedirs("./temp", exist_ok=True)
            save_path = os.path.join("./temp", "dcm_3d.npy")
            np.save(save_path, img_stack)

            return save_path,[info[1] for info in imgs_with_info]

        elif os.path.isfile(path):
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
            spacing = ds.PixelSpacing
            os.makedirs("./temp", exist_ok=True)
            save_path = os.path.join("./temp", "dcm_2d.npy")
            np.save(save_path, img)
            return save_path,[{"PixelSpaceYX":spacing}]

        else:
            return "invalid path"

    except Exception as e:
        return f"failed: {e}"

import numpy as np
import os
from typing import Tuple, List, Dict

def convert_to_mm_physical_space(
    numpy_path: str, 
    info_list: List[Dict]
) -> str:
    """
    将 DICOM 图像转换为以毫米为单位的物理空间，即进行实际的重采样。
    
    参数:
        numpy_path (str): 输入的 numpy 文件路径
        info_list (List[Dict]): 包含 PixelSpacing 和 z 坐标（如果是 3D）的信息列表

    返回:
        str: 重采样后 numpy 文件的保存路径
    """
    # 1. 读取 numpy
    img = np.load(numpy_path)

    # 2. 检查层数与 info_list 一致
    if img.ndim == 3:
        assert img.shape[0] == len(info_list), \
            f"层数不匹配: numpy 有 {img.shape[0]} 层，但 info_list 有 {len(info_list)} 项"
    elif img.ndim == 2:
        assert len(info_list) == 1, \
            f"2D 图像应有 1 个 info，但 info_list 有 {len(info_list)} 项"
    else:
        raise ValueError("只支持 2D 或 3D numpy 图像")

    # 3. 计算缩放因子
    if img.ndim == 3:
        # 获取 spacing
        pixel_spacings = [info["PixelSpaceYX"] for info in info_list]
        z_coords = [info["z"] for info in info_list]

        # 确保所有 PixelSpacing 一致
        first_spacing = pixel_spacings[0]
        for spacing in pixel_spacings:
            assert spacing == first_spacing, "所有层的 PixelSpacing 必须相同"

        y_spacing, x_spacing = first_spacing
        z_diffs = np.diff(z_coords)
        z_spacing = float(np.mean(z_diffs))

        # 原 spacing：[z_spacing, y_spacing, x_spacing]
        # 我们希望变为 1mm spacing，也可改为其他目标 spacing
        target_spacing = [1.0, 1.0, 1.0]  # 可调
        zoom_factors = [
            z_spacing / target_spacing[0],
            y_spacing / target_spacing[1],
            x_spacing / target_spacing[2]
        ]

        print(f"[3D] 原始 spacing: {[z_spacing, y_spacing, x_spacing]}")
        print(f"[3D] zoom_factors: {zoom_factors}")

        resampled_img = zoom(img, zoom=zoom_factors, order=1)  # 使用线性插值

    else:
        # 2D 图像
        y_spacing, x_spacing = info_list[0]["PixelSpaceYX"]

        target_spacing = [1.0, 1.0]  # 目标单位为 1mm
        zoom_factors = [
            y_spacing / target_spacing[0],
            x_spacing / target_spacing[1]
        ]

        print(f"[2D] 原始 spacing: {[y_spacing, x_spacing]}")
        print(f"[2D] zoom_factors: {zoom_factors}")

        resampled_img = zoom(img, zoom=zoom_factors, order=1)

    # 4. 保存
    os.makedirs("./temp", exist_ok=True)
    save_path = os.path.join("./temp", "dcm_resampled_mm.npy")
    np.save(save_path, resampled_img.astype(np.float32))

    return save_path


if __name__ == "__main__":
    print("该文件一般不应用于被执行，除了测试使用")
    path,info = get_dcm_numpy(r"..\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000")
    print(f"保存原文件路径：{path}")
    print(f"元信息0：{info[0]}")
    tansform_path = convert_to_mm_physical_space(path,info)

    img = np.load(tansform_path)
    print(f"转换后文件大小{img.shape}")

    if img.ndim == 3:
        num_slices = img.shape[0]
        print(f"3D Image loaded: {num_slices} slices")
        mid_index = num_slices // 2
        plt.imshow(img[mid_index], cmap="gray")
        plt.title(f"Slice {mid_index}")
        plt.axis("off")
        plt.show()
        yz_slices = img.shape[1]
        mid_index = yz_slices // 2
        plt.imshow(img[:,mid_index,:], cmap="gray")
        plt.title(f"yz Slice {mid_index}")
        plt.axis("off")
        plt.show()
        

