import os
from PIL import Image
import numpy as np

def binarize_image(img_path, save_path, threshold=128):
    img = Image.open(img_path).convert('L')
    arr = np.array(img)
    # 只保留0和255，其余全部归为0或255
    arr = np.where(arr < threshold, 0, 255).astype(np.uint8)
    bin_img = Image.fromarray(arr)
    bin_img.save(save_path)

def binarize_folder(input_folder, output_folder, threshold=128):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            print(f"Binarizing {filename}...")
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)
            binarize_image(in_path, out_path, threshold)

if __name__ == '__main__':
    input_dir = r'D:\MyFile\qq_3045834499\Unet\masks'
    binarize_folder(input_dir, input_dir, threshold=127)  # 你可以根据实际情况调整阈值