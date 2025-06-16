import os
import random
import shutil

def split_dataset(base_dir, seed=42):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    img_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    total = len(img_files)
    print(f"总样本数: {total}")

    # 打乱顺序
    random.seed(seed)
    random.shuffle(img_files)

    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    n_test = total - n_train - n_val

    splits = {
        "train": img_files[:n_train],
        "val": img_files[n_train:n_train + n_val],
        "test": img_files[n_train + n_val:]
    }

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, split), exist_ok=True)
        for img_name in splits[split]:
            label_name = img_name.replace(".jpg", ".txt")
            shutil.move(os.path.join(images_dir, img_name), os.path.join(images_dir, split, img_name))
            shutil.move(os.path.join(labels_dir, label_name), os.path.join(labels_dir, split, label_name))
        print(f"{split} 集: {len(splits[split])} 个样本")

if __name__ == "__main__":
    base_dir = r"D:\MyFile\qq_3045834499\New"
    split_dataset(base_dir)