import os
import numpy as np
import torch
import random


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Preprocessing using preserved HU in dilated part of mask
BASE = 'E:/work_files/praticalTraining_cv/NoduleNet/lbydataAndMidout/' # make sure you have the ending '/'
data_config = {
    # put combined LUNA16 .mhd files into one folder
    'data_dir': BASE + 'subset0',

    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': BASE + 'preprocessed_test_zero/3',

    # put annotation downloaded from LIDC to this path
    'annos_dir': BASE + 'LIDC-XML-only/tcia-lidc-xml',

    # put lung mask downloaded from LUNA16 to this path
    'lung_mask_dir': BASE + 'seg-lungs-LUNA16-zero/',

    # Directory for saving intermediate results
    'ctr_arr_save_dir': BASE + 'annotation/mask_test',
    'mask_save_dir': BASE + 'masks_test',
    'mask_exclude_save_dir': BASE + 'masks_exclude_test',


    'roi_names': ['nodule'],
    'crop_size': [128, 128, 128],
    'bbox_border': 8,
    'pad_value': 170,
    # 'jitter_range': [0, 0, 0],
}


def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
            anchors.append([d, h, w])

    return anchors

bases = [5, 10, 20, 30, 50]
aspect_ratios = [[1, 1, 1]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    'crop_size': data_config['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': [],

    'augtype': {'flip': True, 'rotate': True, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,

    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    # false positive reduction network configuration
    'num_class': len(data_config['roi_names']) + 1,
    'rcnn_crop_size': (7,7,7), # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 32,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'mask_crop_size': [24, 48, 48],
    'mask_test_nms_overlap_threshold': 0.3,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]
    
    
}


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'NoduleNet',
    'batch_size': 4,

    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,

    'epochs': 120,
    'epoch_save': 5,
    'epoch_rcnn': 45,
    'epoch_mask': 60,
    'num_workers': 8,

    'train_set_list': ['split/my/train.csv'],
    'val_set_list': ['split/my/val.csv'],
    'test_set_name': 'split/my/val.csv',
    'label_types': ['mask'],
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3


train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'cross_val_test')
train_config['initial_checkpoint'] = None#train_config['out_dir'] + '/model2/001.ckpt'


config = dict(data_config, **net_config)
config = dict(config, **train_config)
