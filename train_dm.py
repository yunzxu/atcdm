import os
import glob
import torch
import h5py
import numpy as np
from tqdm import tqdm
import yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tool.image_process import normalize_mri
from tool.module.new_dm_module import DM_Inference
from tool.image_process import normalize_mri,normalize_mri_array,load_nifit,resampleVolume
import matplotlib.pyplot as plt
import random
import SimpleITK as sitk
##zoom
from scipy.ndimage import zoom

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

    
setup_seed(42)




def prediction_pixel_DM():
    '''
    用于测试最早训练的pixel DM模型
    '''

    ###load model
    config = {
        'model_config': {
            'UNet': {
                'T': 1000,
                'ch': 32,
                'ch_mult': [1, 2, 2, 2],
                'attn': [1],
                'num_res_blocks': 1,
                'dropout': 0.1,
                'in_ch': 2,
                'out_c': 1,


            },
            ##inference时的参数
            'ddim_steps': 50,
            'num_ddim':5,
            'noiseT':500, ###截断时推理，起始噪声和步数和推理是一致的,500~800
            'infer_T': 50,
            'UNet_checkpoint': '/home/yunzhixu/Project/checkpoint/ASL_LDM/train_SR3DDPM_normadni_GMASL_LRHR_T1000/checkpoint.pth',
            'z_scale': 1.0, # Scale factor for pixel values (if needed)
        },
    }
    pixel_dm_infer = DM_Inference(config=config,
                                  load_fun='state_dict',
                                  patch_size=[96, 96, 96], )
    


    ##load hdf5
    h5_files = '/home/yunzhixu/Data/LDM_Data/ADNI_mask_4mode.hdf5'
    h5f = h5py.File(h5_files,'r')
    T1_data = h5f['T1'][()][151:]
    T1_data = normalize_mri_array(T1_data)
    
    val_data = []
    Hr_data = []
    mask_data = []
    # for val_num in range(151,155):
    #     hdf5data = h5py.File('/home/yunzhixu/Data/LDM_Data/GMASL_data_val/GMASL_ADNIGE_'+str(val_num)+'.hdf5', 'r', libver='latest', swmr=True)
    #     lr_data = hdf5data['ASL_LRNoise'][()]
    #     hr_data = hdf5data['ASL_HR'][()]
    for val_num in range(151,171):
        hdf5data = h5py.File(f'/home/yunzhixu/Data/LDM_Data/val/ADNI_GMWMASL_HRLR_{val_num}.hdf5', 'r', libver='latest', swmr=True)
        print('keys:',list(hdf5data.keys()))
        lr_data = hdf5data['ASL_LR'][()]
        hr_data = hdf5data['ASL_HR'][()]
        mask= hdf5data['ASL_HR'][()]
        mask[mask<0.1] = 0
        mask[mask>0.1] = 1
        mask_data.append(mask)


        lr_data = normalize_mri(lr_data)
        hr_data = normalize_mri(hr_data)
        val_data.append(lr_data)
        Hr_data.append(hr_data)
    val_data = np.array(val_data)
    Hr_data = np.array(Hr_data)
    mask_data = np.array(mask_data)



    ##load nifti
    asl_path ='test'
    ##
    asl_data = load_nifit(asl_path)


    ##如果是原始的ASL 64x3
    ##upsample to :1、如果原始ASL有spacing信息，则使用spacing进行resample；2、如果没有spacing信息，则直接使用插值resize
    asl_img = sitk.ReadImage(asl_path)
    asl_resampled = resampleVolume([1,1,1],asl_img)
    asl_array = sitk.GetArrayFromImage(asl_resampled)
    asl_array -sitk.GetImageFromArray(asl_array)
   

    ##如果没有spacing
    asl_array = zoom(asl_data, (128/asl_data.shape[0],128/asl_data.shape[1],128/asl_data.shape[2]), order=3)


    ##normalize
    asl_data = normalize_mri(asl_data)


    ##T1_data
    t1_data = load_nifit('test')
    t1_data  =normalize_mri(t1_data)

    ##保证t1_data和asl_data shape一致,空间也是一致的

    ##preidction
    recon,metrics = pixel_dm_infer.predict_3d(t1_data, asl_data, patch_size=[96, 96, 96],target=None,mask=None,infer_mode='2-stage',prior_noise='cond')
    '''
    对于predict_3d，第一个输入默认是cond,也就是T1，第二个输入是img，也就是ASL_LR
    prior_noise:设置起始先验噪声是什么：
    'gaussian'：标准高斯噪声
    'cond'：使用cond图像作为起始噪声
    'img'：使用lr_data图像作为起始噪声
    'target':None，不计算指标
    'mask_data：用于计算指标时的mask，None表示不使用mask
    'infer_mode':推理的模式
    输出recon，是矩阵，3D，分布是[-1,1], (reoon+1/2)即为[0,1]范围
    '''







     


if __name__ == "__main__":

    # prediction()
    prediction_pixel_DM()
