import os
import xml.etree.ElementTree as ET
import json
import math

def parse_nodules(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {'ns': 'http://www.nih.gov'}
    nodules = []
    for doctor_id, reading_session in enumerate(root.findall('ns:readingSession', namespace)):
        for nodule in reading_session.findall('ns:unblindedReadNodule', namespace):
            roi = nodule.find('ns:roi', namespace)
            if roi is None:
                continue
            edge_maps = roi.findall('ns:edgeMap', namespace)
            if not edge_maps:
                continue
            # 计算中心点
            xs, ys = [], []
            for edge in edge_maps:
                x = float(edge.find('ns:xCoord', namespace).text)
                y = float(edge.find('ns:yCoord', namespace).text)
                xs.append(x)
                ys.append(y)
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            image_sop_uid = roi.find('ns:imageSOP_UID', namespace).text
            # 恶性程度
            malignancy = None
            characteristics = nodule.find('ns:characteristics', namespace)
            if characteristics is not None:
                malignancy_elem = characteristics.find('ns:malignancy', namespace)
                if malignancy_elem is not None:
                    malignancy = int(malignancy_elem.text)
            nodules.append({
                "doctor_id": doctor_id,
                "center": [center_x, center_y],
                "imageSOP_UID": image_sop_uid,
                "malignancy": malignancy
            })
    return nodules

def euclidean_distance(c1, c2):
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

def cluster_nodules(nodules, distance_threshold=30):
    # 按 imageSOP_UID 分组，每张图片单独聚类
    clusters = []
    for sop_uid in set(n['imageSOP_UID'] for n in nodules):
        nods = [n for n in nodules if n['imageSOP_UID'] == sop_uid]
        grouped = []
        for n in nods:
            found = False
            for group in grouped:
                # 用第一个中心点或均值
                if euclidean_distance(n['center'], group['centers'][0]) < distance_threshold:
                    group['centers'].append(n['center'])
                    group['malignancies'].append(n['malignancy'])
                    found = True
                    break
            if not found:
                grouped.append({
                    "imageSOP_UID": sop_uid,
                    "centers": [n['center']],
                    "malignancies": [n['malignancy']]
                })
        # 聚合每个 group
        for group in grouped:
            avg_center = [
                sum(x) / len(x) for x in zip(*group['centers'])
            ]
            avg_malignancy = (
                sum(m for m in group['malignancies'] if m is not None) /
                len([m for m in group['malignancies'] if m is not None])
                if any(m is not None for m in group['malignancies']) else None
            )
            clusters.append({
                "imageSOP_UID": sop_uid,
                "center": avg_center,
                "malignancy": avg_malignancy
            })
    return clusters

if __name__ == "__main__":
    dicom_folder = r"D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001"
    # 假设 xml 文件在 dicom_folder 下的某个子目录
    xml_file = None
    for root, dirs, files in os.walk(dicom_folder):
        for f in files:
            if f.lower().endswith('.xml'):
                xml_file = os.path.join(root, f)
                break
        if xml_file:
            break
    if not xml_file:
        raise FileNotFoundError("未找到 xml 文件")
    nodules = parse_nodules(xml_file)
    clusters = cluster_nodules(nodules, distance_threshold=10)
    # 输出 json
    output_json = os.path.join(dicom_folder, "nodule_summary.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"已保存结节信息到 {output_json}")