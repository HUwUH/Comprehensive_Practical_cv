from utils.read_file import get_dcm_numpy_with_z
import os
import numpy as np
import pydicom
from matplotlib import pyplot as plt


def get_dcm_numpy(path: str) -> str:
    """
    输入文件路径或文件夹路径，返回 numpy 数组保存路径。
    如果是文件夹，会按 z 坐标排序并堆叠为 3D 数组。
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

            return save_path

        elif os.path.isfile(path):
            ds = pydicom.dcmread(path)
            img = ds.pixel_array
            os.makedirs("./temp", exist_ok=True)
            save_path = os.path.join("./temp", "dcm_2d.npy")
            np.save(save_path, img)
            return save_path

        else:
            return "invalid path"

    except Exception as e:
        return f"failed: {e}"

if __name__ == "__main__":
    print("该文件一般不应用于被执行，除了测试使用")
    print(get_dcm_numpy(r"E:\realtrain\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000"))
    npy_path = "./temp/dcm_3d.npy"
    img = np.load(npy_path)
    if img.ndim == 3:
        num_slices = img.shape[0]
        print(f"3D Image loaded: {num_slices} slices")
        mid_index = num_slices // 2
        plt.imshow(img[mid_index], cmap="gray")
        plt.title(f"Slice {mid_index}")
        plt.axis("off")
        plt.show()
        

