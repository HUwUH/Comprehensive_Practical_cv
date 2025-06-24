import os
import json
import pydicom
import numpy as np
import cv2

def dcm_to_jpg(dcm_path, jpg_path, window_center=-600, window_width=1600):
    dcm = pydicom.dcmread(dcm_path)
    pixel_data = dcm.pixel_array.astype(np.float32)
    slope = getattr(dcm, 'RescaleSlope', 1)
    intercept = getattr(dcm, 'RescaleIntercept', 0)
    hu = pixel_data * slope + intercept
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2
    windowed = np.clip(hu, window_min, window_max)
    windowed = ((windowed - window_min) / (window_max - window_min) * 255).astype(np.uint8)
    cv2.imwrite(jpg_path, windowed)
    return windowed.shape  # 返回图片尺寸

def convert_to_yolo(center, r, img_w, img_h):
    x, y = center
    x_center = x / img_w
    y_center = y / img_h
    w = h = 2 * r
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def process_all_cases(root_dir, out_dir):
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    idx = 0
    for case_folder in os.listdir(root_dir):
        print(f"Processing case: {case_folder}")
        case_path = os.path.join(root_dir, case_folder)
        if not (case_folder.startswith("LIDC-IDRI-") and os.path.isdir(case_path)):
            continue
        summary_json = os.path.join(case_path, "nodule_summary_new.json")
        if not os.path.exists(summary_json):
            continue
        with open(summary_json, "r", encoding="utf-8") as f:
            nodules = json.load(f)
        file_groups = {}
        for nodule in nodules:
            filename = nodule["filename"]
            if filename is None:
                continue
            center = nodule["center"]
            malignancy = nodule.get("malignancy", None)
            r = 10 if malignancy is None else 10 + 3 * float(malignancy)
            if filename not in file_groups:
                file_groups[filename] = []
            file_groups[filename].append((center, r))
        for filename, centers in file_groups.items():
            dicom_path = os.path.join(case_path, filename)
            if not os.path.exists(dicom_path):
                continue
            img_name = f"{idx:06d}.jpg"
            label_name = f"{idx:06d}.txt"
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, label_name)
            img_h, img_w = dcm_to_jpg(dicom_path, img_path, window_center=-600, window_width=1600)
            with open(label_path, "w") as f:
                for center, r in centers:
                    x_center, y_center, w_norm, h_norm = convert_to_yolo(center, r, img_w, img_h)
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            idx += 1

if __name__ == "__main__":
    root_dir = r"D:\MyFile\LIDC-IDRI"
    out_dir = r"D:\MyFile\qq_3045834499\NewMoreJPG"
    process_all_cases(root_dir, out_dir)