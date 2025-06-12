import pydicom
import matplotlib.pyplot as plt
import numpy as np
from typing import NoReturn


def get_dcm_numpy_with_z(paths:list[str])->list[tuple[np.ndarray,dict]]:
    """
    读取多个 DICOM 文件，返回其（像素array，info）的列表。
    """
    imgs = []
    for path in paths:
        ds = pydicom.dcmread(path)

        spacing = ds.PixelSpacing
        z = float(ds.ImagePositionPatient[2])
        info = {"z":z,"PixelSpaceYX":spacing}

        imgs.append((ds.pixel_array,info))
    return imgs

def plt_show_dcm_numpy(array:np.ndarray)->NoReturn:
    """
    显示灰度图。
    """
    assert len(array.shape)==2,"please one channel"
    plt.imshow(array, cmap='gray')
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()


# 读取 DICOM 文件
if __name__ == "__main__":
    print("这个文件不该被运行，运行仅为了测试功能")
    dcm_file = r"E:\work_files\praticalTraining_cv\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000\000002.dcm"
    img,info = get_dcm_numpy_with_z([dcm_file])[0]
    plt_show_dcm_numpy(img)
    print(info)