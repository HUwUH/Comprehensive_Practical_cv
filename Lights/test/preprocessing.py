import os
import xml.etree.ElementTree as ET
import json
import math

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
            roi = nodule.find('ns:roi', namespace)
            if roi is None:
                continue
            edge_maps = roi.findall('ns:edgeMap', namespace)
            if not edge_maps:
                continue
            xs, ys = [], []
            for edge in edge_maps:
                x = float(edge.find('ns:xCoord', namespace).text)
                y = float(edge.find('ns:yCoord', namespace).text)
                xs.append(x)
                ys.append(y)
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
                "center": [center_x, center_y],
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
                    group['malignancies'].append(n['malignancy'])
                    found = True
                    break
            if not found:
                grouped.append({
                    "imageSOP_UID": sop_uid,
                    "filename": n['filename'],
                    "centers": [n['center']],
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
            clusters.append({
                "imageSOP_UID": sop_uid,
                "filename": group['filename'],
                "center": avg_center,
                "malignancy": avg_malignancy
            })
    return clusters

if __name__ == "__main__":
    a = 1
    root_dir = r"D:\MyFile\LIDC-IDRI"
    for case_folder in os.listdir(root_dir):
        # 跳过前1006个文件夹
        if a <= 1006:
            a += 1
            continue
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
            print(f"跳过: {case_folder}（缺少DICOM或XML）")
            continue
        nodules = parse_nodules(xml_file, sopuid_to_filename)
        clusters = cluster_nodules(nodules, distance_threshold=10)
        output_json = os.path.join(case_path, "nodule_summary.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(clusters, f, ensure_ascii=False, indent=2)
        print(f"已保存: {output_json}")