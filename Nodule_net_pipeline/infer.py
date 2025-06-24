import os
import numpy as np
import torch
import SimpleITK as sitk
import math
import nrrd
import torch.nn.functional as F
# from .light_dataset import pad2factor
# from .config import config
# from .net.nodule_net import NoduleNet
# from .utils.util import crop_boxes2mask_single

from light_dataset import pad2factor
from config import config
from net.nodule_net import NoduleNet
from utils.util import crop_boxes2mask_single

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)  # shape: [D, H, W]
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, origin, spacing

def preprocess_image(img):
    # pad到16的倍数
    img_pad = pad2factor(img, factor=16, pad_value=0)
    # 归一化，跟训练时一致
    img_norm = (img_pad.astype(np.float32) - 128) / 128
    # 增加channel维度，变成(1, D, H, W)
    img_norm = np.expand_dims(img_norm, axis=0)
    return img_norm

def inference_single_image(model, img_path, device):
    model.eval()
    model.use_rcnn = True
    model.use_mask = True

    # 读图
    img, origin, spacing = load_itk_image(img_path)
    img_proc = preprocess_image(img)

    input_tensor = torch.from_numpy(img_proc).float().to(device).unsqueeze(0)  # batch=1

    # 推理
    with torch.no_grad():
        # 这里推理时没有标注数据，传None即可
        model.forward(input_tensor, truth_boxes=None, truth_labels=None, truth_masks=None, masks=None)

    segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in model.mask_probs] if model.mask_probs else []
    pred_mask = crop_boxes2mask_single(model.crop_boxes[:, 1:], segments, img.shape)
    pred_mask = pred_mask.astype(np.uint8)
    # TODO:修改这里


    # # 获得检测框与掩码
    # crop_boxes = model.crop_boxes  # numpy格式的整数坐标框，shape [N, 8]
    # mask_probs = model.mask_probs.cpu().numpy() if len(model.mask_probs) else []
    # # 原图尺寸
    # img_reso = img.shape
    # # 将局部掩码映射到原图3D掩码
    # pred_mask = crop_boxes2mask_single(crop_boxes[:, 1:-1], mask_probs, img_reso)

    # 额外得到检测框
    detections = model.detections.cpu().numpy() if hasattr(model, 'detections') else None

    return detections, pred_mask, origin, spacing, img_proc

def save_results(out_dir, pid, detections, mask):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'{pid}_detections.npy'), detections)
    np.save(os.path.join(out_dir, f'{pid}_mask.npy'), mask)

def main_inference(model_weight_path, test_img_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型结构与权重
    # config = {...}  # 你的配置字典
    model = NoduleNet(config, mode='eval')
    model.to(device)

    checkpoint = torch.load(model_weight_path)
    model.load_state_dict(checkpoint['state_dict'])

    # 单张推理
    pid = os.path.basename(test_img_path).replace('.mhd', '')
    detections, mask, origin, spacing, img_proc = inference_single_image(model, test_img_path, device)

    # 保存结果
    save_results(output_dir, pid, img_proc[0], mask)
    print(img_proc[0].shape)
    print(f'Inference done for {pid}, results saved to {output_dir}')

if __name__ == '__main__':
    model_weight_path = 'E:\work_files/praticalTraining_cv/NoduleNet/results/cross_val_test/model/100.ckpt'
    test_img_path = 'E:\work_files/praticalTraining_cv/Comprehensive_Practical_cv/Nodule_net_pipeline/test\processed\processed_clean.nrrd'
    output_dir = 'E:\work_files/praticalTraining_cv/Comprehensive_Practical_cv/Nodule_net_pipeline/test/inference_results'

    main_inference(model_weight_path, test_img_path, output_dir)
