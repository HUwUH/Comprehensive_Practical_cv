import os
import json
import pydicom
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np
import time

def dcm_to_pil_image(dcm_path, window_center=-600, window_width=1600):
    dcm = pydicom.dcmread(dcm_path)
    pixel_data = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, 'RescaleSlope', 1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    hu = pixel_data * slope + intercept
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    windowed = np.clip(hu, window_min, window_max)
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    return Image.fromarray(windowed).convert("RGB")

def draw_nodules_on_image(dicom_path, centers, output_path, window_center=-600, window_width=1600):
    img = dcm_to_pil_image(dicom_path, window_center, window_width)
    draw = ImageDraw.Draw(img)
    color_map = {1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "purple"}
    for center, malignancy in centers:
        x, y = center
        if malignancy is None:
            color = "blue"
            r = 10
        else:
            color = color_map.get(int(np.floor(malignancy)), "blue")
            r = 10 + 3 * float(malignancy)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    img.save(output_path)

if __name__ == "__main__":
    root_dir = r"D:\MyFile\LIDC-IDRI"
    all_null_filenames = []
    a = 1
    for case_folder in os.listdir(root_dir):
        # 跳过前 125
        if a <= 760:
            a += 1
            continue
        case_path = os.path.join(root_dir, case_folder)
        if not (case_folder.startswith("LIDC-IDRI-") and os.path.isdir(case_path)):
            continue
        summary_json = os.path.join(case_path, "nodule_summary_new.json")
        while not os.path.exists(summary_json):
            print(f"等待: {case_folder}（无nodule_summary.json），正在等待文件生成...")
            time.sleep(1)
        output_folder = os.path.join(case_path, "nodule_center_vis")
        os.makedirs(output_folder, exist_ok=True)
        with open(summary_json, "r", encoding="utf-8") as f:
            nodules = json.load(f)
        file_groups = defaultdict(list)
        for nodule in nodules:
            filename = nodule["filename"]
            if filename is None:
                nodule_with_case = dict(nodule)
                nodule_with_case["case_folder"] = case_folder
                all_null_filenames.append(nodule_with_case)
                continue
            center = nodule["center"]
            malignancy = nodule.get("malignancy", None)
            file_groups[filename].append((center, malignancy))
        for idx, (filename, centers) in enumerate(file_groups.items(), 1):
            dicom_path = os.path.join(case_path, filename)
            if not os.path.exists(dicom_path):
                print(f"未找到DICOM文件: {dicom_path}")
                continue
            output_path = os.path.join(output_folder, f"New_image_{idx}_{filename}.png")
            draw_nodules_on_image(dicom_path, centers, output_path)
            print(f"已保存: {output_path}")
    if all_null_filenames:
        print("所有病例中 filename 为 None 的结节如下：")
        for n in all_null_filenames:
            print(n)
        log_path = os.path.join(os.getcwd(), "nodule_null_filename.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            for n in all_null_filenames:
                logf.write(json.dumps(n, ensure_ascii=False) + "\n")
        print(f"已保存 log 文件: {log_path}")