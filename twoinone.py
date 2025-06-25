import numpy as np
import SimpleITK as sitk

def apply_red_mask(img, mask, label_mask=None):
    """
    将三维灰度图像转换为三通道图像，mask > 0 的部分设为红色，label_mask > 0 的部分设为黄色，
    两者的交集设为橙色，其余保持灰度

    参数:
        img (numpy.ndarray): 输入灰度图像，形状 (D, H, W) ，为int8
        mask (numpy.ndarray): 掩码矩阵，形状与 img 一致，值类型为整数
        label_mask (numpy.ndarray): 掩码矩阵，形状与 img 一致，值类型为整数

    返回:
        numpy.ndarray: 三通道图像，形状 (D, H, W, 3)
    """
    img = np.load(img)
    mask = np.load(mask)
    if label_mask is not None:
        itkimage = sitk.ReadImage(label_mask)
        label_mask = sitk.GetArrayFromImage(itkimage)

    # 检查输入形状是否一致
    assert img.shape == mask.shape, f"img 和 mask 的形状不一致: img {img.shape}, mask {mask.shape}"
    if label_mask is not None:
        assert img.shape == label_mask.shape, f"img 和 label_mask 的形状不一致: img {img.shape}, label_mask {label_mask.shape}"

    # 创建三通道输出图像
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3)).astype(np.uint8)

    # 先将图像转为灰度三通道
    gray = img
    output[..., 0] = gray  # 红色通道
    output[..., 1] = gray  # 绿色通道
    output[..., 2] = gray  # 蓝色通道

    # 处理label_mask和mask的交集（橙色）优先度最高
    if label_mask is not None:
        intersection = (mask > 0) & (label_mask > 0)
        output[intersection, 0] = 255  # 红色通道
        output[intersection, 1] = 165  # 绿色通道 (橙色的RGB值为255,165,0)
        output[intersection, 2] = 0  # 蓝色通道

    # 处理mask > 0的部分（红色），但要排除已经处理过的交集部分
    red_mask = (mask > 0)
    if label_mask is not None:
        red_mask = red_mask & ~intersection  # 排除交集部分
    output[red_mask, 0] = 255  # 红色通道
    output[red_mask, 1] = 0  # 绿色通道
    output[red_mask, 2] = 0  # 蓝色通道

    # 处理label_mask > 0的部分（黄色），但要排除已经处理过的交集部分
    if label_mask is not None:
        yellow_mask = (label_mask > 0)
        if label_mask is not None:
            yellow_mask = yellow_mask & ~intersection  # 排除交集部分
        output[yellow_mask, 0] = 255  # 红色通道
        output[yellow_mask, 1] = 255  # 绿色通道 (黄色的RGB值为255,255,0)
        output[yellow_mask, 2] = 0  # 蓝色通道

    np.save(r'D:\PycharmProjects\Comprehensive_Practical_cv\test\output\ans', output)

    return output

if __name__ == '__main__':
    out = apply_red_mask(r'D:\PycharmProjects\Comprehensive_Practical_cv\test\output\processed_clean.nrrd_detections.npy',
                   r'D:\PycharmProjects\Comprehensive_Practical_cv\test\output\processed_clean.nrrd_mask.npy')
    import matplotlib.pyplot as plt
    plt.imshow(out[:,:,200])
    plt.show()