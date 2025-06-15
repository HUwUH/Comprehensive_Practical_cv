import os
import json
import pydicom
from PIL import Image, ImageDraw
from collections import defaultdict
import numpy as np

def draw_nodules_on_image(dicom_path, centers, output_path):
    ds = pydicom.dcmread(dicom_path)
    img = Image.fromarray(ds.pixel_array).convert("RGB")
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
        # 跳过前1006个文件夹
        if a <= 1006:
            a += 1
            continue
        case_path = os.path.join(root_dir, case_folder)
        if not (case_folder.startswith("LIDC-IDRI-") and os.path.isdir(case_path)):
            continue
        summary_json = os.path.join(case_path, "nodule_summary.json")
        if not os.path.exists(summary_json):
            print(f"跳过: {case_folder}（无nodule_summary.json）")
            continue
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
            output_path = os.path.join(output_folder, f"image_{idx}_{filename}.png")
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