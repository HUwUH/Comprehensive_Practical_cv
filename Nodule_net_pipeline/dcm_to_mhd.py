import os
import SimpleITK as sitk
import shutil

def convert_dcm_to_mhd(dcm_folder, output_path):
    """
    选择一个存有dcm文件的文件夹，整理出outputpath的mhd文件到指定文件路径
    在前端使用的时候，要注意，为了下一个函数的调用，output_path所在的文件夹应该干净
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_folder)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    sitk.WriteImage(image, output_path, useCompression=True)

# 示例用法
# convert_dcm_to_mhd(r'E:\work_files\praticalTraining_cv\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000', './patient001.mhd')
if __name__ == "__main__":
    '''将整个数据集转换为mhd格式'''
    confirm = input("你知道自己在运行什么吗？这可能会造成很大的硬盘开销")
    if confirm != "yes":
        print("正在退出")
        exit(0)
    
    target_mhd_path = "../../MYLIDC/mhd"
    target_xml_path = "../../MYLIDC/xml"
    os.makedirs(target_mhd_path,exist_ok=True)
    os.makedirs(target_xml_path,exist_ok=True)
    base_path = "../../LIDC-IDRI"

    if (not os.path.isdir(base_path)) or len(os.listdir(base_path))==0:
        assert False,"文件夹不对"
    for patient_fold in os.listdir(base_path): # 访问病人
        patient_path = os.path.join(base_path,patient_fold)
        patient_id = patient_fold[-4:]
        if not os.path.isdir(patient_path): 
            print(f"病人同等深度路径被访问，但不是文件夹：{patient_path}")
            continue

        study_info = []
        for study_fold in os.listdir(patient_path): # 访问片子id
            study_path = os.path.join(patient_path, study_fold)
            study_id = study_fold
            series_folder = os.path.join(study_path, "000000")

            if not os.path.isdir(study_path):
                print(f"study同等深度路径被访问，但不是文件夹：{study_path}")
                continue
            if not os.path.isdir(series_folder):
                print(f"study同等深度路径被访问，但不包含000000：{series_folder}，")
                continue

            file_count = len(os.listdir(series_folder))
            study_info.append((file_count, study_id))
        
        study_info.sort(key=lambda x: x[0], reverse=True)
        if not study_info:
            print(f"病人不包含任何有效study：{patient_id}")
            continue
        elif study_info[0][0]<10:
            print(f"病人的study疑似过短：{patient_id}")
            continue
        else:
            study_original_path = os.path.join(patient_path, study_info[0][1],'000000')
            study_target_xml_path = os.path.join(target_xml_path, "patient"+patient_id+".xml")
            study_target_mhd_path = os.path.join(target_mhd_path, "patient"+patient_id+".mhd")
            study_original_xml_path = None
            for file in os.listdir(study_original_path):
                if os.path.isfile(os.path.join(study_original_path,file)) and file.endswith(".xml"):
                    study_original_xml_path = os.path.join(study_original_path,file)
            
            if not study_original_xml_path:
                print(f"病人的最长study没有xml：{study_original_path}")
                continue
            # 标注
            shutil.copy(study_original_xml_path,study_target_xml_path)
            # 文件夹
            convert_dcm_to_mhd(study_original_path,study_target_mhd_path)
