import os
import xml.etree.ElementTree as ET
import json
import math
import numpy as np
from PIL import Image, ImageDraw
import pydicom
import cv2

def dcm_to_jpg_array(dcm_path, window_center=-600, window_width=1600):
    """
    将DICOM文件转换为适合保存的numpy数组
    Args:
        dcm_path: DICOM文件路径
        window_center: 窗位
        window_width: 窗宽
    Returns:
        numpy array: 处理后的图像数组
    """
    # 读取 DICOM 文件
    dcm = pydicom.dcmread(dcm_path)
    # 获取像素数据
    pixel_data = dcm.pixel_array.astype(np.float32)
    # 获取 DICOM 文件中的斜率和截距
    slope = dcm.RescaleSlope if hasattr(dcm, 'RescaleSlope') else 1
    intercept = dcm.RescaleIntercept if hasattr(dcm, 'RescaleIntercept') else 0
    # 应用斜率和截距进行线性变换
    hu = pixel_data * slope + intercept

    # 计算窗位和窗宽的上下限
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # 进行窗宽窗位的调整
    windowed = np.clip(hu, window_min, window_max)
    # 归一化到 0-255 范围
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    
    return windowed

def build_sopuid_to_filename_map(dicom_folder):
    sopuid_to_filename = {}
    for root, dirs, files in os.walk(dicom_folder):
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                try:
                    import pydicom
                    ds = pydicom.dcmread(path, stop_before_pixels=True)
                    sopuid_to_filename[ds.SOPInstanceUID] = f
                except Exception:
                    continue
    return sopuid_to_filename

def parse_nodules(xml_file, sopuid_to_filename):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {'ns': 'http://www.nih.gov'}
    nodules = []

    for doctor_id, reading_session in enumerate(root.findall('ns:readingSession', namespace)):
        for nodule in reading_session.findall('ns:unblindedReadNodule', namespace):
            for roi in nodule.findall('ns:roi', namespace):
                image_zposition = roi.find('ns:imageZposition', namespace)
                if image_zposition is None:
                    continue
                image_zposition = image_zposition.text
                edge_maps = roi.findall('ns:edgeMap', namespace)
                if not edge_maps:
                    continue
                xs, ys = [], []
                edge_points = []
                for edge in edge_maps:
                    x = float(edge.find('ns:xCoord', namespace).text)
                    y = float(edge.find('ns:yCoord', namespace).text)
                    xs.append(x)
                    ys.append(y)
                    edge_points.append([x, y])
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)
                image_sop_uid = roi.find('ns:imageSOP_UID', namespace).text
                filename = sopuid_to_filename.get(image_sop_uid)
                malignancy = None
                characteristics = nodule.find('ns:characteristics', namespace)
                if characteristics is not None:
                    malignancy_elem = characteristics.find('ns:malignancy', namespace)
                    if malignancy_elem is not None:
                        malignancy = int(malignancy_elem.text)
                nodules.append({
                    "doctor_id": doctor_id,
                    "imageZposition": image_zposition,
                    "center": [center_x, center_y],
                    "edge_points": edge_points,
                    "imageSOP_UID": image_sop_uid,
                    "filename": filename,
                    "malignancy": malignancy
                })
    return nodules

def euclidean_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def cluster_nodules(nodules, distance_threshold=30):
    clusters = []
    for sop_uid in set(n['imageSOP_UID'] for n in nodules):
        nods = [n for n in nodules if n['imageSOP_UID'] == sop_uid]
        grouped = []
        for n in nods:
            found = False
            for group in grouped:
                if euclidean_distance(n['center'], group['centers'][0]) < distance_threshold:
                    group['centers'].append(n['center'])
                    group['edge_points_list'].append(n['edge_points'])
                    group['malignancies'].append(n['malignancy'])
                    found = True
                    break
            if not found:
                grouped.append({
                    "imageSOP_UID": sop_uid,
                    "imageZposition": n['imageZposition'],
                    "filename": n['filename'],
                    "centers": [n['center']],
                    "edge_points_list": [n['edge_points']],
                    "malignancies": [n['malignancy']]
                })
        for group in grouped:
            avg_center = [
                sum(x) / len(x) for x in zip(*group['centers'])
            ]
            avg_malignancy = (
                sum(m for m in group['malignancies'] if m is not None) /
                len([m for m in group['malignancies'] if m is not None])
                if any(m is not None for m in group['malignancies']) else None
            )
            # 合并所有边界点
            all_edge_points = []
            for edge_points in group['edge_points_list']:
                all_edge_points.extend(edge_points)
            
            clusters.append({
                "imageSOP_UID": sop_uid,
                "imageZposition": group['imageZposition'],
                "filename": group['filename'],
                "center": avg_center,
                "edge_points": all_edge_points,
                "malignancy": avg_malignancy
            })
    return clusters

def optimize_edge_points_order(edge_points):
    """
    优化边界点的顺序，确保它们形成合理的轮廓
    Args:
        edge_points: 原始边界点列表 [[x1, y1], [x2, y2], ...]
    Returns:
        list: 优化后的边界点列表
    """
    if len(edge_points) < 3:
        return edge_points
    
    try:
        # 计算质心
        center_x = sum(p[0] for p in edge_points) / len(edge_points)
        center_y = sum(p[1] for p in edge_points) / len(edge_points)
        
        # 根据每个点相对于质心的角度对点进行排序
        def angle_from_center(point):
            return math.atan2(point[1] - center_y, point[0] - center_x)
        
        sorted_points = sorted(edge_points, key=angle_from_center)
        return sorted_points
    except:
        # 如果角度排序失败，返回原始点列表
        return edge_points

def create_mask_from_points(edge_points, image_size=(512, 512)):
    """
    根据边界点创建蒙版（改进版本，处理所有数量的边界点）
    Args:
        edge_points: 边界点列表 [[x1, y1], [x2, y2], ...]
        image_size: 图像尺寸 (width, height)
    Returns:
        numpy array: 二值蒙版，结节区域为255，背景为0
    """
    # 创建空白蒙版
    mask = np.zeros(image_size[::-1], dtype=np.uint8)  # 注意：OpenCV使用(height, width)
    
    if len(edge_points) == 0:
        return mask
    
    # 处理只有1个或2个点的情况
    if len(edge_points) == 1:
        # 单点：创建一个小圆形区域（半径为3像素）
        x, y = edge_points[0]
        x, y = int(round(x)), int(round(y))
        if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
            cv2.circle(mask, (x, y), 3, 255, -1)
        return mask
    
    if len(edge_points) == 2:
        # 两点：创建连接两点的线段，并加粗
        x1, y1 = edge_points[0]
        x2, y2 = edge_points[1]
        x1, y1 = int(round(x1)), int(round(y1))
        x2, y2 = int(round(x2)), int(round(y2))
        # 绘制粗线段
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=5)
        # 在端点添加圆形
        cv2.circle(mask, (x1, y1), 3, 255, -1)
        cv2.circle(mask, (x2, y2), 3, 255, -1)
        return mask
    
    # 优化边界点顺序
    edge_points = optimize_edge_points_order(edge_points)
    
    # 预处理边界点：去除重复点并确保点的有效性
    unique_points = []
    for x, y in edge_points:
        # 确保坐标在图像范围内
        x = max(0, min(image_size[0] - 1, x))
        y = max(0, min(image_size[1] - 1, y))
        rounded_point = (int(round(x)), int(round(y)))
        # 避免连续重复点
        if not unique_points or unique_points[-1] != rounded_point:
            unique_points.append(rounded_point)
    if len(unique_points) < 3:
        return mask
    
    # 使用高级多边形填充方法，保持原始形状
    mask = advanced_polygon_fill(unique_points, image_size)
    
    # 最终确保蒙版是严格的二值化图像
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    return mask

def create_combined_mask_from_multiple_nodules(nodules_list, image_size=(512, 512)):
    """
    为同一图像中的多个结节创建合并的蒙版（改进版本）
    Args:
        nodules_list: 同一图像中所有结节的边界点列表
        image_size: 图像尺寸 (width, height)    
        Returns:
        numpy array: 二值蒙版，所有结节区域为255，背景为0
    """
    # 创建空白蒙版
    combined_mask = np.zeros(image_size[::-1], dtype=np.uint8)
    
    for edge_points in nodules_list:
        if len(edge_points) > 0:  # 只要有边界点就处理
            # 为每个结节单独创建蒙版
            single_mask = create_mask_from_points(edge_points, image_size)
            # 使用逻辑或操作合并蒙版
            combined_mask = cv2.bitwise_or(combined_mask, single_mask)
      # 最后的形态学操作来确保蒙版质量
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 确保蒙版是严格的二值化图像（0或255）
    combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    return combined_mask

def load_nodule_summary(json_path):
    """
    加载nodule_summary_new.json文件
    Args:
        json_path: JSON文件路径
    Returns:
        dict: 按filename分组的结节信息
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            nodules = json.load(f)
        
        # 按filename分组
        grouped_by_filename = {}
        for nodule in nodules:
            filename = nodule.get('filename')
            if filename:
                if filename not in grouped_by_filename:
                    grouped_by_filename[filename] = []
                grouped_by_filename[filename].append(nodule)
        
        return grouped_by_filename
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return {}

def save_masks_for_unet_optimized(dicom_folder, output_folder, start_index=0):
    """
    为U-Net训练保存蒙版图像（优化版本：使用JSON文件，每个DICOM文件只处理一次）
    Args:
        dicom_folder: DICOM文件夹路径
        output_folder: 输出文件夹路径
        start_index: 起始索引
    Returns:
        int: 下一个可用的索引
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建images和masks子文件夹
    images_folder = os.path.join(output_folder, 'images')
    masks_folder = os.path.join(output_folder, 'masks')
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(masks_folder, exist_ok=True)
    
    # 查找JSON文件
    json_path = os.path.join(dicom_folder, "nodule_summary_new.json")
    if not os.path.exists(json_path):
        print(f"未找到JSON文件: {json_path}")
        return start_index
    
    # 加载并按filename分组结节信息
    grouped_nodules = load_nodule_summary(json_path)
    if not grouped_nodules:
        return start_index
    
    current_index = start_index
    
    # 处理每个唯一的DICOM文件
    for filename, nodules_in_file in grouped_nodules.items():
        # 查找DICOM文件路径
        dicom_path = None
        for root, dirs, files in os.walk(dicom_folder):
            if filename in files:
                dicom_path = os.path.join(root, filename)
                break
        
        if not dicom_path:
            continue
        
        try:
            # 使用dcm_to_jpg_array函数处理DICOM图像
            image_array = dcm_to_jpg_array(dicom_path)
            
            # 生成六位数字的文件名
            image_filename = f"{current_index:06d}.jpg"
            mask_filename = f"{current_index:06d}.jpg"            
            # 保存原始图像
            cv2.imwrite(os.path.join(images_folder, image_filename), image_array)
            
            # 收集该文件中所有结节的边界点
            all_edge_points_list = []
            for nodule in nodules_in_file:
                edge_points = nodule.get('edge_points', [])
                if edge_points and len(edge_points) > 0:  # 只要有边界点就处理
                    all_edge_points_list.append(edge_points)
            
            # 创建合并的蒙版
            if all_edge_points_list:
                combined_mask = create_combined_mask_from_multiple_nodules(
                    all_edge_points_list, 
                    image_array.shape[::-1]
                )
                # 保存蒙版
                cv2.imwrite(os.path.join(masks_folder, mask_filename), combined_mask)
                
                nodule_count = len(all_edge_points_list)
                print(f"已保存: {current_index:06d}.jpg (包含{nodule_count}个结节)")
            else:
                # 如果没有有效的边界点，创建空蒙版
                empty_mask = np.zeros(image_array.shape[::-1], dtype=np.uint8)
                cv2.imwrite(os.path.join(masks_folder, mask_filename), empty_mask)
                print(f"已保存: {current_index:06d}.jpg (无有效结节)")
            
            current_index += 1
            
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")
            continue
    
    return current_index

def advanced_polygon_fill(edge_points, image_size):
    """
    高级多边形填充，保持原始形状的同时处理空洞
    Args:
        edge_points: 边界点列表
        image_size: 图像尺寸 (width, height)
    Returns:
        numpy array: 填充后的蒙版
    """
    mask = np.zeros(image_size[::-1], dtype=np.uint8)
    
    if len(edge_points) < 3:
        return mask
    
    # 转换为numpy数组并确保在图像范围内
    points = []
    for x, y in edge_points:
        x = max(0, min(image_size[0] - 1, int(round(x))))
        y = max(0, min(image_size[1] - 1, int(round(y))))
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    # 直接填充多边形
    cv2.fillPoly(mask, [points], 255)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        # 重新绘制最大的轮廓（主要结节）
        mask.fill(0)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [largest_contour], 255)
        # 非常轻微的形态学处理，只填充1-2像素的小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 确保蒙版是严格的二值化图像（0或255）
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    return mask

if __name__ == "__main__":
    # 设置Unet输出文件夹
    unet_train_folder = r"D:\MyFile\qq_3045834499\Unet"
    # 全局索引计数器，确保所有文件名唯一
    global_index = 0
    
    root_dir = r"D:\MyFile\LIDC-IDRI"
    for case_folder in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case_folder)
        if not (case_folder.startswith("LIDC-IDRI-") and os.path.isdir(case_path)):
            continue
        print(f"处理: {case_folder}")
        
        # 查找DICOM和XML
        dicom_folder = case_path
        sopuid_to_filename = build_sopuid_to_filename_map(dicom_folder)
        xml_file = None
        for root, dirs, files in os.walk(case_path):
            for f in files:
                if f.lower().endswith('.xml'):
                    xml_file = os.path.join(root, f)
                    break
            if xml_file:
                break
        if not xml_file or not sopuid_to_filename:
            continue
            
        # 首先生成JSON文件（如果不存在）
        json_path = os.path.join(case_path, "nodule_summary_new.json")
        if not os.path.exists(json_path):
            nodules = parse_nodules(xml_file, sopuid_to_filename)
            clusters = cluster_nodules(nodules, distance_threshold=10)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(clusters, f, ensure_ascii=False, indent=2)
        
        # 使用优化的函数保存蒙版（直接从JSON文件读取）
        global_index = save_masks_for_unet_optimized(dicom_folder, unet_train_folder, global_index)
        
    print(f"完成！总共处理了 {global_index} 个图像，保存到: {unet_train_folder}")
