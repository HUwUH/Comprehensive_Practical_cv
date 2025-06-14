import os
import xml.etree.ElementTree as ET
import pydicom
from PIL import Image, ImageDraw
from collections import defaultdict

# 预定义颜色列表
DOCTOR_COLORS = [
    "red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"
]

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {'ns': 'http://www.nih.gov'}
    nodules = []
    for doctor_id, reading_session in enumerate(root.findall('ns:readingSession', namespace)):
        for nodule in reading_session.findall('ns:unblindedReadNodule', namespace):
            roi = nodule.find('ns:roi', namespace)
            image_zposition = roi.find('ns:imageZposition', namespace).text
            image_sop_uid = roi.find('ns:imageSOP_UID', namespace).text
            edge_map = []
            for edge in roi.findall('ns:edgeMap', namespace):
                x = float(edge.find('ns:xCoord', namespace).text)
                y = float(edge.find('ns:yCoord', namespace).text)
                edge_map.append((x, y))
            nodules.append({
                "imageZposition": image_zposition,
                "imageSOP_UID": image_sop_uid,
                "edgeMap": edge_map,
                "doctor_id": doctor_id
            })
    return nodules

def find_dicom_file(dicom_folder, sop_uid):
    for file_name in os.listdir(dicom_folder):
        if file_name.endswith('.dcm'):
            dicom_file = pydicom.dcmread(os.path.join(dicom_folder, file_name))
            if dicom_file.SOPInstanceUID == sop_uid:
                print(f"找到匹配的 DICOM 文件: {file_name} (SOP UID: {sop_uid})")
                return dicom_file
    return None

def draw_contours_on_image(dicom_file, edge_maps, output_path):
    pixel_array = dicom_file.pixel_array
    image = Image.fromarray(pixel_array).convert("RGB")
    draw = ImageDraw.Draw(image)
    for edge_map, doctor_id in edge_maps:
        color = DOCTOR_COLORS[doctor_id % len(DOCTOR_COLORS)]
        if len(edge_map) <= 1:
            if edge_map:
                x, y = edge_map[0]
                r = 8
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        else:
            for i in range(len(edge_map) - 1):
                x1, y1 = edge_map[i]
                x2, y2 = edge_map[i + 1]
                draw.line((x1, y1, x2, y2), fill=color, width=2)
    image.save(output_path)

def process_nodules(xml_file, dicom_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    nodules = parse_xml(xml_file)
    if not nodules:
        print("未找到任何结节信息")
        return
    # 按 imageSOP_UID 分组
    nodule_groups = defaultdict(list)
    for nodule in nodules:
        nodule_groups[nodule["imageSOP_UID"]].append((nodule["edgeMap"], nodule["doctor_id"]))
    for idx, (sop_uid, edge_maps) in enumerate(nodule_groups.items(), 1):
        dicom_file = find_dicom_file(dicom_folder, sop_uid)
        if dicom_file is None:
            print(f"未找到与 SOP UID {sop_uid} 匹配的 DICOM 文件")
            continue
        output_path = os.path.join(output_folder, f"image_{idx}_sop_{sop_uid}.png")
        draw_contours_on_image(dicom_file, edge_maps, output_path)
        print(f"已保存图片 {output_path}，包含 {len(edge_maps)} 个结节")

if __name__ == "__main__":
    dicom_folder = r"D:\MyFile\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000"
    output_folder = "nodule_contours"
    xml_files = [f for f in os.listdir(dicom_folder) if f.lower().endswith('.xml')]
    if len(xml_files) != 1:
        raise FileNotFoundError("DICOM 文件夹中必须且只能有一个 .xml 文件")
    xml_file = os.path.join(dicom_folder, xml_files[0])
    print(f"找到 XML 文件: {xml_file}")
    process_nodules(xml_file, dicom_folder, output_folder)