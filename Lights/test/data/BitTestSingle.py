import os
import pydicom
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

root_dir = r'D:\MyFile\LIDC-IDRI'

def process_case(case_folder, shared_set):
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
    shared_set.update(local_set)
    print(f"已完成检查: {case_folder}")

if __name__ == "__main__":
    manager = Manager()
    shared_set = manager.set()
    case_folders = [f for f in os.listdir(root_dir)
                    if f.startswith("LIDC-IDRI-") and os.path.isdir(os.path.join(root_dir, f))]
    with ProcessPoolExecutor(max_workers=20) as executor:
        for case_folder in case_folders:
            executor.submit(process_case, case_folder, shared_set)

    result = sorted(list(shared_set))
    print("所有出现过的 (BitsAllocated, BitsStored) 组合：")
    print(result)