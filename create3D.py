import numpy as np
from skimage import measure
import trimesh


def npy_to_3d_model(npy_path, output_path, threshold=None):
    # 加载.npy文件
    data = np.load(npy_path)

    # 处理不同形状和数据类型
    if data.ndim == 4 and data.shape[-1] == 3:  # (420,420,350,3) 彩色体素
        # 转换为灰度用于表面提取
        grayscale = np.mean(data, axis=-1).astype(np.float32)
    elif data.ndim == 3:  # (332,360,360) 或 (420,420,350)
        grayscale = data.astype(np.float32)
    else:
        raise ValueError("Unsupported array shape")

    # 自动确定阈值（如果未提供）
    if threshold is None:
        threshold = np.percentile(grayscale, 50)  # 取中值

    # 使用Marching Cubes提取网格
    verts, faces, normals, _ = measure.marching_cubes(
        grayscale,
        level=threshold,
        spacing=(1., 1., 1.),  # 调整此参数以匹配真实物理尺寸
        allow_degenerate=False
    )

    # 创建网格对象
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # 保存为3D模型文件（支持.stl/.obj/.ply）
    mesh.export(output_path)


# 示例使用
npy_to_3d_model(r'D:\PycharmProjects\Comprehensive_Practical_cv\test\output\ans.npy', r'D:\PycharmProjects\Comprehensive_Practical_cv\show\model.stl', threshold=128)  # 彩色数据
npy_to_3d_model(r'D:\PycharmProjects\Comprehensive_Practical_cv\temp\dcm_resampled_mm.npy', r'D:\PycharmProjects\Comprehensive_Practical_cv\show\model.obj', threshold=0.5)  # float32数据
npy_to_3d_model(r'D:\PycharmProjects\Comprehensive_Practical_cv\test\output\processed_clean.nrrd_detections.npy', r'D:\PycharmProjects\Comprehensive_Practical_cv\show\model.ply')  # 自动阈值