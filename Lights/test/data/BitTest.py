import os
import pydicom
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

root_dir = r'D:\MyFile\LIDC-IDRI'
bits_set = set()
lock = threading.Lock()

def process_case(case_folder):
    local_set = set()
    case_path = os.path.join(root_dir, case_folder)
    for file in os.listdir(case_path):
        if not file.lower().endswith('.dcm'):
            continue
        dcm_path = os.path.join(case_path, file)
        try:
            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            bits_alloc = getattr(dcm, "BitsAllocated", None)
            bits_stored = getattr(dcm, "BitsStored", None)
            if bits_alloc is not None and bits_stored is not None:
                local_set.add((bits_alloc, bits_stored))
        except Exception as e:
            print(f"读取失败: {dcm_path}，原因: {e}")
    with lock:
        bits_set.update(local_set)
    print(f"已完成检查: {case_folder}")

if __name__ == "__main__":
    case_folders = [f for f in os.listdir(root_dir)
                    if f.startswith("LIDC-IDRI-") and os.path.isdir(os.path.join(root_dir, f))]
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_case, case_folder) for case_folder in case_folders]
        for future in as_completed(futures):
            pass  # 等待所有任务完成

    result = sorted(list(bits_set))
    print("所有出现过的 (BitsAllocated, BitsStored) 组合：")
    print(result)