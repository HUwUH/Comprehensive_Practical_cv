import os
import shutil
import xml.etree.ElementTree as ET
import logging

# 配置日志记录
logging.basicConfig(filename='lidc_idri_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 定义 LIDC - IDRI 数据集的根目录
root_dir = r'D:\MyFile\LIDC-IDRI'

# 遍历根目录下的所有 LIDC - IDRI - XXXX 文件夹
for lidc_idri_folder in os.listdir(root_dir):
    if not lidc_idri_folder.startswith("LIDC-IDRI-") or not os.path.isdir(os.path.join(root_dir, lidc_idri_folder)):
        continue

    # 获取 LIDC - IDRI - XXXX 文件夹的路径
    lidc_idri_path = os.path.join(root_dir, lidc_idri_folder)

    # 遍历 LIDC - IDRI - XXXX 文件夹下的所有子文件夹
    for sub_folder in os.listdir(lidc_idri_path):
        sub_folder_path = os.path.join(lidc_idri_path, sub_folder)

        # 如果子文件夹不是目录，跳过
        if not os.path.isdir(sub_folder_path):
            continue

        # 查找以 "000000" 命名的文件夹
        for sub_sub_folder in os.listdir(sub_folder_path):
            if sub_sub_folder == "000000":
                data_folder_path = os.path.join(sub_folder_path, sub_sub_folder)

                # 查找唯一的 .xml 文件
                xml_files = [f for f in os.listdir(data_folder_path) if f.endswith(".xml")]
                if len(xml_files) != 1:
                    logging.warning(f"在 {data_folder_path} 中未找到或找到多个 XML 文件，跳过此文件夹。")
                    continue

                xml_file_path = os.path.join(data_folder_path, xml_files[0])

                # 解析 XML 文件
                try:
                    tree = ET.parse(xml_file_path)
                    root = tree.getroot()
                    namespace = {'ns': 'http://www.nih.gov'}

                    # 查找 ResponseHeader 字段
                    response_header = root.find('ns:ResponseHeader', namespace)
                    if response_header is None:
                        logging.warning(f"在 XML 文件 {xml_file_path} 中未找到 ResponseHeader 字段，跳过此文件夹。")
                        continue

                    # 查找 TaskDescription 字段
                    task_description = response_header.find("ns:TaskDescription", namespace)
                    if task_description is None:
                        logging.warning(f"在 XML 文件 {xml_file_path} 中未找到 TaskDescription 字段，跳过此文件夹。")
                        continue

                    task_description_text = task_description.text

                    # 如果是 X 光数据，删除整个子文件夹
                    if task_description_text == "CXR read":
                        logging.info(f"删除 X 光数据文件夹: {sub_folder_path}")
                        shutil.rmtree(sub_folder_path)
                        break

                    # 如果是 CT 数据，将 "000000" 文件夹中的数据移动到 LIDC - IDRI - XXXX 文件夹中
                    elif task_description_text == "Second unblinded read":
                        logging.info(f"移动 CT 数据文件夹: {data_folder_path} -> {lidc_idri_path}")
                        for item in os.listdir(data_folder_path):
                            # 如果文件已经存在于目标路径中，跳过
                            if os.path.exists(os.path.join(lidc_idri_path, item)):
                                logging.warning(f"文件 {item} 已存在于 {lidc_idri_path}，跳过移动。")
                                continue
                            shutil.move(os.path.join(data_folder_path, item), lidc_idri_path)

                        # 删除空的 "000000" 文件夹
                        # os.rmdir(data_folder_path)

                        # 删除当前子文件夹（sub_folder）
                        logging.info(f"删除子文件夹: {sub_folder_path}")
                        shutil.rmtree(sub_folder_path)
                        break

                except ET.ParseError as e:
                    logging.error(f"解析 XML 文件 {xml_file_path} 时出错: {e}")

    # 清理 LIDC - IDRI - XXXX 文件夹中的所有子文件夹
    for item in os.listdir(lidc_idri_path):
        item_path = os.path.join(lidc_idri_path, item)
        if os.path.isdir(item_path):
            logging.info(f"删除子文件夹: {item_path}")
            shutil.rmtree(item_path)

    # 删除 AccuImage.dir 文件
    for file_name in os.listdir(lidc_idri_path):
        if file_name == "AccuImage.dir":
            os.remove(os.path.join(lidc_idri_path, file_name))
            logging.info(f"删除文件: {os.path.join(lidc_idri_path, file_name)}")

    logging.info(f"处理完成: {lidc_idri_folder}")

logging.info("所有文件夹处理完成！")