from . import config as config
from .dcm_to_mhd import convert_dcm_to_mhd
from .generate_all_dummy_lung_mask import generate_dummy_lung_mask
from .preprocess import preprocess_for_inference
from .infer import main_inference

import os
from os import path as path

def preprocess_pipeline(input_folder:str,
             mhd_save_folder:str,
             dummy_lungmask_folder:str,
             preprocessed_folder:str)->str:
    """
    返回success，表示成功
    返回其他，表示失败
    此外，有raise的可能性
    """
    # 0.创建文件夹
    os.makedirs(mhd_save_folder,exist_ok=True)
    if len(os.listdir(mhd_save_folder))!=0:
        return "finished0 mhd_save_folder应当为空文件夹"
    os.makedirs(dummy_lungmask_folder,exist_ok=True)
    os.makedirs(preprocessed_folder,exist_ok=True)
    
    # 1.将dcm文件，变成mhd文件（压缩）
    mhdfile_name = "temp_patient"
    mhdfile_path = path.join(mhd_save_folder,mhdfile_name+".mhd")
    convert_dcm_to_mhd(input_folder,mhdfile_path)
    # 2.创建虚拟掩码
    generate_dummy_lung_mask(mhdfile_path,dummy_lungmask_folder)
    lungmaskfile_path = path.join(dummy_lungmask_folder,mhdfile_name+".mhd")
    # 3.预处理肺部图像
    preprocess_for_inference(mhdfile_path,lungmaskfile_path,preprocessed_folder)


if __name__ == "__main__":
    from dcm_to_mhd import convert_dcm_to_mhd
    from generate_all_dummy_lung_mask import generate_dummy_lung_mask
    from preprocess import preprocess_for_inference
    # from infer import main_inference
    fd_input = r"E:\work_files\praticalTraining_cv\LIDC-IDRI\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\000000"
    fd1 = "./test/mhd"
    fd2 = "./test/lung"
    fd3 = "./test/processed"
    fd4 = "./test/output"

    print(preprocess_pipeline(fd_input,fd1,fd2,fd3))
    model_weight_path = 'E:\work_files/praticalTraining_cv/NoduleNet/results/cross_val_test/model/100.ckpt'
    main_inference(model_weight_path, path.join(fd3,"processed_clean.nrrd"), fd4)