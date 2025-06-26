import SimpleITK as sitk
import numpy as np
import os
from glob import glob

def generate_dummy_lung_mask(img_path, save_path, label_value=3):
    itk_img = sitk.ReadImage(img_path)
    np_img = sitk.GetArrayFromImage(itk_img)

    dummy_mask = np.ones_like(np_img, dtype=np.uint8) * label_value
    dummy_itk = sitk.GetImageFromArray(dummy_mask)

    # 保持与原始图像一致的元信息
    dummy_itk.CopyInformation(itk_img)

    pid = os.path.splitext(os.path.basename(img_path))[0]
    save_file = os.path.join(save_path, f"{pid}.mhd")
    sitk.WriteImage(dummy_itk, save_file, useCompression=True)
    print(f"[✔] Saved dummy lung mask: {save_file}")


def generate_all_dummy_lung_masks(img_dir, lung_mask_dir, label_value=3):
    if not os.path.exists(lung_mask_dir):
        os.makedirs(lung_mask_dir)

    img_files = glob(os.path.join(img_dir, "*.mhd"))

    if not img_files:
        print("[!] No .mhd files found in img_dir. Please check the path.")
        return

    for img_path in img_files:
        generate_dummy_lung_mask(img_path, lung_mask_dir, label_value)

    print(f"\n[✅] Done. Generated {len(img_files)} dummy lung masks.")


# 🔧 修改以下路径为你的实际路径
if __name__ == "__main__":
    img_dir = "./subset0"  # 原始CT图像的目录
    lung_mask_dir = "E:/work_files/praticalTraining_cv/NoduleNet/lbydataAndMidout/seg-lungs-LUNA16-zero/"  # 生成的dummy掩码保存目录

    generate_all_dummy_lung_masks(img_dir, lung_mask_dir, label_value=3)
