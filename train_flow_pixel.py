import os
import glob
import torch
import h5py
import numpy as np
import yaml
import random
import matplotlib.pyplot as plt
import time
from tool.module.flow_module_pixel import FM_Pixel_Module, Pixel_Inference,evaluate_metric
from tool.image_process import normalize_mri, normalize_mri_array, save_nifit, display_center_slices,crop_pad3D,load_nifit
from tool import load_yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_default_config_pixel():
    """Return default configuration for Pixel Space ASL Flow Matching"""
    train_name = 'train_I2SBPixelFM_v5_HighnoiseGMASL_ch32v8attn8_earlystop_7500'
    #'train_PixelFM_finetuneMyASL_1e5_from_HighnoiseGMASL_ch32v8attn8_earlystop_7500'
    return {
        "use_lora": False,
        'flow_matching_type':'I2SB',
        'model_config': {
            'Type': 'pixel',
            'flow_matching_type':'I2SB',
            'UNet': {
                "dims": 3,
                "image_size": 96,
                # in_channels = Target(1) + Condition(1) = 2
                "in_channels": 2, 
                "out_channels": 1, 
                "model_channels": 32,
                "attention_resolutions": [8],
                "num_res_blocks": 2,
                "channel_mult": [1, 2, 4, 8],
                "num_head_channels": 8,
                "context_dim": None,
                "use_spatial_transformer": False,
            },
            'UNet_checkpoint':None,#'/home/yunzhixu/Project/hydro_jupiter/train_I2SBPixelFM_v3_HighnoiseGMASL_ch32v8attn8_earlystop_7500/checkpoint.pth',
            #/home/yunzhixu/Project/hydro_jupiter/train_FM_HighnoiseGMASL_lre5_base_Pretrain_GMASL_Pixel_p96_controlt1w_scale1_lrsche10000/checkpoint.pth',
            'z_scale': 1.0, # Scale factor for pixel values (if needed)
        },
        'train_config': {
            'lr': 1e-4,
            'lr_scheduler': 8000,
            'log_dir': '/home/yunzhixu/Project/hydro_jupiter/source/run/' + train_name,
            'train_name': train_name,
            'root_path': '/home/yunzhixu/Project/hydro_jupiter/',
            'save_dir': '/home/yunzhixu/Project/hydro_jupiter/' + train_name,
            'save_interval':500,
            'val_interval': 200,
        },
        'data_config': {
            'train_paths': ['/home/yunzhixu/Data/LDM_Data/ADNI_GMWMASL_HRLR_individual/train/'],#['/home/yunzhixu/Data/LDM_Data/MyASL/train/'],
            'val_paths': ['/home/yunzhixu/Data/LDM_Data/ADNI_GMWMASL_HRLR_individual/val/'],
            # Corresponds to [Target, Condition]
            'get_data_name': ['ASL_HR', 'ASL_LR'],
            'all_data': False,
            'batch_size': 2,
            'patch_size': [96, 96, 96],
            'hot_label': False,
            'patch_pro': 0.7,
            'add_noise_tensor': False, ##随机给条件输入加噪声
        }
    }



# def get_default_config_pixel():
#     """Return default configuration for Pixel Space ASL Flow Matching"""
#     train_name = 'train_PixelFM_PhysicalGMASL_ch32v8attn8'
#     #'train_PixelFM_finetuneMyASL_1e5_from_HighnoiseGMASL_ch32v8attn8_earlystop_7500'
#     return {
#         "use_lora": False,
#         # 'flow_matching_type':'I2SB',
#         'model_config': {
#             'Type': 'pixel',
#             'UNet': {
#                 "dims": 3,
#                 "image_size": 96,
#                 # in_channels = Target(1) + Condition(1) = 2
#                 "in_channels": 2, 
#                 "out_channels": 1, 
#                 "model_channels": 32,
#                 "attention_resolutions": [8],
#                 "num_res_blocks": 2,
#                 "channel_mult": [1, 2, 4, 8],
#                 "num_head_channels": 8,
#                 "context_dim": None,
#                 "use_spatial_transformer": False,
#             },
#             'UNet_checkpoint':None,#'/home/yunzhixu/Project/hydro_jupiter/train_PixelFM_HighnoiseGMASL_ch32v8attn8_earlystop_7500/checkpoint.pth',
#             #/home/yunzhixu/Project/hydro_jupiter/train_FM_HighnoiseGMASL_lre5_base_Pretrain_GMASL_Pixel_p96_controlt1w_scale1_lrsche10000/checkpoint.pth',
#             'z_scale': 1.0, # Scale factor for pixel values (if needed)
#         },
#         'train_config': {
#             'lr': 1e-4,
#             'lr_scheduler': 8000,
#             'log_dir': '/home/yunzhixu/Project/hydro_jupiter/source/run/' + train_name,
#             'train_name': train_name,
#             'root_path': '/home/yunzhixu/Project/hydro_jupiter/',
#             'save_dir': '/home/yunzhixu/Project/hydro_jupiter/' + train_name,
#             'save_interval':500,
#             'val_interval': 200,
#         },
#         'data_config': {
#             'train_paths': ['/home/yunzhixu//Data/adni_ge/physics_based_ASL_h5/train/'],#['/home/yunzhixu/Data/LDM_Data/MyASL/train/'],
#             'val_paths': ['/home/yunzhixu//Data/adni_ge/physics_based_ASL_h5/val/'],
#             # Corresponds to [Target, Condition]
#             'get_data_name': ['ASL_HR', 'ASL_LR'],
#             'all_data': False,
#             'batch_size': 2,
#             'patch_size': [96, 96, 96],
#             'hot_label': False,
#             'patch_pro': 0.7,
#             'add_noise_tensor': False, ##随机给条件输入加噪声
#         }
#     }

def train_GMASL_pixel():
    '''
    Train Pixel-Space Flow Matching on Synthetic ASL data.
    '''
    config = get_default_config_pixel()
    model = FM_Pixel_Module(config).to(device)
    
    print(f"Starting training for: {config['train_config']['train_name']}")
    model.train_loop(total_steps=100000)


def predict_GMASL_pixel(model_name = 'train_PixelFM_HighnoiseGMASL_ch32v8attn8',
                        flow_matching_type = 'SR3',
                        save_file_name ='GMASL',
                        fm_num_steps = 50,
                        file_name_prefix = 'GMASL_HighNoise_recon_50steps',
                        prior_noise = 'Gaussian'):
    '''
    Validation/Prediction on Synthetic ASL Validation data (Pixel Space).
    '''
    val_data = []
    target_data = []
    mask_data = []

    T1_data = []
    h5_files = '/home/yunzhixu/Data/LDM_Data/ADNI_mask_4mode.hdf5'
    h5f = h5py.File(h5_files,'r')
    T1_data = h5f['T1'][()][151:]
    T1_data = normalize_mri_array(T1_data)
    
    # Load validation samples
    for val_num in range(151, 171): # Adjusted range for quick test
        # file_path = f'/home/yunzhixu/Data/LDM_Data/GMASL_data_val/GMASL_ADNIGE_{val_num}.hdf5'
        ##High Noise Validation Set
        file_path = f'/home/yunzhixu/Data/LDM_Data/ADNI_GMWMASL_HRLR_individual/val/ADNI_GMWMASL_HRLR_{val_num}.hdf5'
        if not os.path.exists(file_path):
            continue
            
        hdf5data = h5py.File(file_path, 'r', libver='latest', swmr=True)
        print('key', list(hdf5data.keys()))
        lr_data = hdf5data['ASL_LR'][()]
        hr_data = hdf5data['ASL_HR'][()]
        # T1_data = hdf5data['T1'][()]
        mask_tmp = hr_data.copy()
        mask_data.append(mask_tmp)

        lr_data = normalize_mri(lr_data)
        hr_data = normalize_mri(hr_data)
        
        lr_data = np.nan_to_num(lr_data, nan=-1)
        hr_data = np.nan_to_num(hr_data, nan=-1)
        
        target_data.append(hr_data)
        val_data.append(lr_data)
    
    if not val_data:
        print("No validation data found.")
        return

    val_data = np.array(val_data)
    target_data = np.array(target_data)
    mask_data = np.array(mask_data)

    # mask_data = T1_data.copy()
    mask_data[mask_data < 0.1] = 0
    mask_data[mask_data >0.1] = 1

    # Load Config
    # model_name = 'train_I2SBPixelFM_HighnoiseGMASL_ch32v8attn8_earlystop_7500_continue'

    # Update checkpoint path manually or dynamically
    config = load_yaml('/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/config.yml')
    config['model_config']['UNet_checkpoint'] = '/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/checkpoint.pth'
    save_dir = '/home/yunzhixu/Project/hydro_jupiter/' + model_name +'/'+ save_file_name
    os.makedirs(save_dir, exist_ok=True)
    # Inference
    pixel_infer = Pixel_Inference(config, device=device,flow_matching_type=flow_matching_type)
    ssim_list = []
    psnr_list = []

    for i in range(len(val_data)):
        print(f"Running inference on {len(val_data)} samples...")
        time0 = time.time()
        if prior_noise == 'T1w':
            #T1w prior
            print('T1_data[i] shape:',T1_data[i].shape)
            recon,metrics = pixel_infer.predict_3d(T1_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=fm_num_steps,prior_noise='T1w',noise_T=800)
            # recon,metrics = pixel_infer.predict_3d(T1_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=fm_num_steps,prior_noise='T1w')
        # recon,metrics = pixel_infer.predict_3d(T1_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=10,prior_noise='T1w',noise_T=800)
        #No t1w
        elif prior_noise == 'Gaussian':
            recon,metrics = pixel_infer.predict_3d(val_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=fm_num_steps)
        else:
            raise ValueError("Invalid prior_noise type. Choose 'T1w' or 'Gaussian'.")
        save_nifit(recon, os.path.join(save_dir, f'{file_name_prefix}_{i+151}.nii.gz'))
        time1 = time.time()
        print(f"Sample {i} inference time: {time1 - time0} seconds")
    
        ssim_list.append(metrics[0])
        psnr_list.append(metrics[1])
    print('len(ssim_list):',len(ssim_list))
    print('len(psnr_list):',len(psnr_list))
    ssim_list = np.array(ssim_list)

    psnr_list = np.array(psnr_list)
    print('SSIM List:', np.round(ssim_list,4))
    print('PSNR List:', np.round(psnr_list,4))
    print('Average SSIM:', np.round(np.mean(ssim_list),4),'±',np.round(np.std(ssim_list),4))
    print('Average PSNR:', np.round(np.mean(psnr_list),4),'±',np.round(np.std(psnr_list),4))
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write('Reconstruction Metrics\n')
        f.write('Sample\tSSIM\tPSNR\n')
        for i in range(len(ssim_list)):
            f.write(f'{i}\t{ssim_list[i]:.4f}\t{psnr_list[i]:.4f}\n')
        #写入mean
        f.write(f'Average SSIM: {np.round(np.mean(ssim_list),4)} ± {np.round(np.std(ssim_list),4)}\n')
        f.write(f'Average PSNR: {np.round(np.mean(psnr_list),4)} ± {np.round(np.std(psnr_list),4)}\n')

        # Save Results
        # os.makedirs(save_dir, exist_ok=True)
        # display_center_slices(recon, title="Reconstructed Pixel Flow", save_path=os.path.join(save_dir, 'A_recon_ASL_Pixel.png'))
        # display_center_slices(val_data[0], title="Original LR Input", save_path=os.path.join(save_dir, 'A_LR_original.png'))
        # display_center_slices(target_data[0], title="Original HR Target", save_path=os.path.join(save_dir, 'A_HR_original.png'))
        
        # save_nifit(recon, os.path.join(save_dir, 'recon_pixel_sample.nii.gz'))


def predict_MyASL_pixel(model_name = 'train_PixelFM_HighnoiseGMASL_ch32v8attn8_earlystop_7500',
                        flow_matching_type = 'SR3',
                        save_file_name ='MyASL_from_LPM',
                        fm_num_steps = 5,
                        file_name_prefix = 'MyASL_HighNoise_recon_5steps',
                        prior_noise = 'Gaussian',
                        myasl_sr = 'MyASL'):
    '''
    Prediction on Real ASL Data (Pixel Space).
    myasl_sr: 'MyASL' or 'MyASL_SLR3' ##测试不同的MyASL数据集
    '''
    file_path = '/home/yunzhixu/Data/LDM_Data/'+myasl_sr+'.hdf5'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    hdf5data = h5py.File(file_path, 'r', libver='latest', swmr=True)
    ASL_data = hdf5data['ASL_LR'][()]
    ASL_HR  = hdf5data['ASL_HR'][()]
    T1_data = hdf5data['T1'][()]
    mask_data = T1_data.copy()
    
    ASL_data = normalize_mri_array(ASL_data)
    ASL_HR = normalize_mri_array(ASL_HR)
    T1_data = normalize_mri_array(T1_data)  

    # model_name = 'train_PixelFM_HighnoiseGMASL_ch32v8attn8_earlystop_7500'
    # Update checkpoint path manually or dynamically
    config = load_yaml('/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/config.yml')
    config['model_config']['UNet_checkpoint'] = '/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/checkpoint.pth'
    save_dir = '/home/yunzhixu/Project/hydro_jupiter/' + model_name +'/'+ save_file_name
    os.makedirs(save_dir, exist_ok=True)
    # Inference
    pixel_infer = Pixel_Inference(config, device=device,flow_matching_type=flow_matching_type)
    


    ssim_list = []
    psnr_list = []


    for i in range(ASL_data.shape[0]): # Limit to 5 samples

        asl_input = load_nifit('/home/yunzhixu/Project/hydro_jupiter/train_FM_HighnoiseGMASL_ch96_zscale1_basevae_train_klVAE_ch64v4_p96_from_GMASL_data_base_t1w_resampled_HRpro08_kl1_le4/MyASL/MyASL_HighNoise_recon_10steps_{}.nii.gz'.format(i))
        print(f"Processing MyASL sample {i}...")
        mask_tmp = mask_data[i]
        mask_tmp[mask_tmp < 1] = 0
        mask_tmp[mask_tmp >= 1] = 1
        print(f"mask_tmp",mask_tmp.shape)
        t1_img = T1_data[i]
        # asl_input  = ASL_data[i]
        # asl_input = Less_min(asl_input)
        
        # Standard LR -> HR inference (Gaussian start)
        final_recon = 0
        # for step in range(5):  # Average over 5 runs for stability
        #     recon = pixel_infer.predict_3d(asl_input, asl_input, patch_size=[96, 96, 96], target=None,num_steps=20)
        #     final_recon += recon
        # recon = final_recon / 5.0
        recon,metrics = pixel_infer.predict_3d(asl_input, asl_input, patch_size=[96, 96, 96], target=ASL_HR[i],mask=mask_tmp,num_steps=50)

        if prior_noise == 'T1w':
            #T1w prior
            recon,metrics = pixel_infer.predict_3d(t1_img, asl_input,patch_size=[96, 96, 96], target=ASL_HR[i],mask=mask_data[i],num_steps=fm_num_steps,prior_noise='T1w',noise_T=800)
        #No t1w
        elif prior_noise == 'Gaussian':
            recon,metrics = pixel_infer.predict_3d(asl_input, asl_input,patch_size=[96, 96, 96], target=ASL_HR[i],mask=mask_data[i],num_steps=fm_num_steps)
        else:
            raise ValueError("Invalid prior_noise type. Choose 'T1w' or 'Gaussian'.")


        print(f"Metrics for sample {i}: SSIM={metrics[0]}, PSNR={metrics[1]}")
        ##查看是否已经存在
        if os.path.exists(os.path.join(save_dir, f'{file_name_prefix}_{i}.nii.gz')):
            print(f"File {file_name_prefix}_{i}.nii.gz already exists. Skipping save.")
            continue
        save_nifit(recon, os.path.join(save_dir, f'{file_name_prefix}_{i}.nii.gz'))
        ssim_list.append(metrics[0])
        psnr_list.append(metrics[1])
    ssim_list = np.array(ssim_list)
    psnr_list = np.array(psnr_list)
    print('MyASL Average SSIM:', np.round(np.mean(ssim_list),4),'±',np.round(np.std(ssim_list),4))
    print('MyASL Average PSNR:', np.round(np.mean(psnr_list),4),'±',np.round(np.std(psnr_list),4))
    ##把结果写入txt
    with open(os.path.join(save_dir, 'MyASL_Pixel_metrics.txt'), 'w') as f:
        f.write('MyASL Pixel Flow Reconstruction Metrics\n')
        f.write('Sample\tSSIM\tPSNR\n')
        for i in range(len(ssim_list)):
            f.write(f'{i}\t{ssim_list[i]:.4f}\t{psnr_list[i]:.4f}\n')
        #写入mean
        f.write(f'Average SSIM: {np.round(np.mean(ssim_list),4)} ± {np.round(np.std(ssim_list),4)}\n')
        f.write(f'Average PSNR: {np.round(np.mean(psnr_list),4)} ± {np.round(np.std(psnr_list),4)}\n')
        
        # save_nifit(recon, os.path.join(output_dir, f'MyASL_recon_{i}.nii.gz'))
        # display_center_slices(recon, title=f"Reconstructed Flow {i}", save_path=os.path.join(output_dir, f'A_recon_MyASL_{i}.png'))
        # display_center_slices(ASL_data[i], title=f"Input LR ASL {i}", save_path=os.path.join(output_dir, f'A_input_MyASL_LR_{i}.png'))
        # display_center_slices(ASL_HR[i], title=f"Target HR ASL {i}", save_path=os.path.join(output_dir, f'A_target_MyASL_HR_{i}.png'))
        
        # Note: 'prior' (starting from T1) is not standard in Pixel Flow Matching (Gaussian->Data).
        # We skip T1-prior inference here unless OT-Flow specifically trained for Image->Image is implemented.
        # recon = pixel_infer.predict_3d(T1_data[i], ASL_data[i], patch_size=[96, 96, 96], target=ASL_HR[i])
        # save_nifit(recon, os.path.join(output_dir, f'MyASL_recon_T1start_{i}.nii.gz'))
        # display_center_slices(recon, title=f"Reconstructed Flow T1 Start {i}", save_path=os.path.join(output_dir, f'recon_MyASL_T1start_{i}.png'))



def Less_min(ASL_data):
    '''
    去除配准带来的背景噪声影响
    '''
    array = ASL_data
    h,bin = np.histogram(a=array.flatten(), bins=50)
    index = np.where(h>0)[0][1] +2
    mins = bin[index]
    print('mins',mins)


    new_data = array -mins
    new_data[new_data<=0] =0
    return new_data



def evaluate_test():
    '''
    Evaluate SSIM and PSNR between reconstructed and target images within the mask.
    '''
    val_data = []
    target_data = []
    mask_data = []
    
    # Load validation samples
    for val_num in range(151, 171): # Adjusted range for quick test
        # file_path = f'/home/yunzhixu/Data/LDM_Data/GMASL_data_val/GMASL_ADNIGE_{val_num}.hdf5'
        ##High Noise Validation Set
        file_path = f'/home/yunzhixu/Data/LDM_Data/ADNI_GMWMASL_HRLR_individual/val/ADNI_GMWMASL_HRLR_{val_num}.hdf5'
        if not os.path.exists(file_path):
            continue
            
        hdf5data = h5py.File(file_path, 'r', libver='latest', swmr=True)
        print('key', list(hdf5data.keys()))
        lr_data = hdf5data['ASL_LR'][()]
        hr_data = hdf5data['ASL_HR'][()]
        # T1_data = hdf5data['T1'][()]
        mask_tmp = hr_data.copy()
        mask_data.append(mask_tmp)

        lr_data = normalize_mri(lr_data)
        hr_data = normalize_mri(hr_data)
        
        lr_data = np.nan_to_num(lr_data, nan=-1)
        hr_data = np.nan_to_num(hr_data, nan=-1)
        
        target_data.append(hr_data)
        val_data.append(lr_data)
    
    if not val_data:
        print("No validation data found.")
        return

    val_data = np.array(val_data)
    target_data = np.array(target_data)
    mask_data = np.array(mask_data) 
    

    # mask_data = T1_data.copy()
    mask_data[mask_data < 0.1] = 0
    mask_data[mask_data >0.1] = 1

    # mask_data = T1_data.copy()
    mask_data[mask_data < 0.1] = 0
    mask_data[mask_data >0.1] = 1


    ssim_list = []
    psnr_list = []
    pred_path = '/home/yunzhixu/Project/hydro_jupiter/train_PixelFM_HighnoiseGMASL_ch32v8attn8/GMASL/'
    for i in range(len(val_data)):
        recon = load_nifit(os.path.join(pred_path, f'GMASL_HighNoise_recon_{i+151}.nii.gz'))
        mask = mask_data[i]
        target = target_data[i]
        print('shape recon:',recon.shape,' shape target:',target.shape,' shape mask:',mask.shape)
        target_tensor = torch.tensor(target[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32).to(device)
        reconstruction = torch.tensor(recon[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(mask[np.newaxis, np.newaxis, :, :, :], dtype=torch.float32).to(device)

        ##归一化：
        target_tensor = (target_tensor + 1.0) / 2.0


        metrics = evaluate_metric(target_tensor, reconstruction,mask=mask_tensor)  
        print(f"Image Metrics : SSIM: {metrics[0]}, PSNR: {metrics[1]}")
        ssim_list.append(metrics[0])
        psnr_list.append(metrics[1])
    ssim_list = np.array(ssim_list)
    psnr_list = np.array(psnr_list)
    print('Average SSIM:', np.round(np.mean(ssim_list),4),'±',np.round(np.std(ssim_list),4))
    print('Average PSNR:', np.round(np.mean(psnr_list),4),'±',np.round(np.std(psnr_list),4))




from tool.module.flow_module_pixel import Pixel_Inference_Guided
def prediction_pixel_guide(model_name = 'train_PixelFM_HighnoiseGMASL_ch32v8attn8',
                        save_file_name ='GMASL_GuidedT1w',
                        fm_num_steps = 10,
                        file_name_prefix = 'GMASL_HighNoise_recon_10steps'):
    '''
    Prediction with Pixel Space Flow Matching with guidance.
    '''
    val_data,target_data, mask_data ,T1_data= [],[],[],[]

    h5_files = '/home/yunzhixu/Data/LDM_Data/ADNI_mask_4mode.hdf5'
    T1_data = h5py.File(h5_files,'r')['T1'][()][151:]
    mask_data = T1_data.copy()
    mask_data[mask_data < 0.1] = 0
    mask_data[mask_data >= 0.1] = 1
    T1_data = normalize_mri_array(T1_data)
    
    # Load validation samples
    for val_num in range(151, 171): # Adjusted range for quick test

        file_path = f'/home/yunzhixu/Data/LDM_Data/ADNI_GMWMASL_HRLR_individual/val/ADNI_GMWMASL_HRLR_{val_num}.hdf5'
        hdf5data = h5py.File(file_path, 'r', libver='latest', swmr=True)

        lr_data = hdf5data['ASL_LR'][()]
        hr_data = hdf5data['ASL_HR'][()]
        lr_data = normalize_mri(lr_data)
        hr_data = normalize_mri(hr_data)
        lr_data = np.nan_to_num(lr_data, nan=-1)
        hr_data = np.nan_to_num(hr_data, nan=-1)
        target_data.append(hr_data)
        val_data.append(lr_data)
    val_data = np.array(val_data)
    target_data = np.array(target_data)

    ##parameters


    config = load_yaml('/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/config.yml')
    config['model_config']['UNet_checkpoint'] = '/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/checkpoint.pth'
    save_dir = '/home/yunzhixu/Project/hydro_jupiter/' + model_name +'/'+ save_file_name
    os.makedirs(save_dir, exist_ok=True)

    pixel_infer = Pixel_Inference_Guided(config, device=device)
    ssim_list = []
    psnr_list = []
    for i in range(len(val_data)):
        print(f"Running guided inference on {len(val_data)} samples...")
        time0 = time.time()
        recon,metrics = pixel_infer.predict_guided_3d(T1_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=fm_num_steps)
        save_nifit(recon, os.path.join(save_dir, f'{file_name_prefix}_{i+151}.nii.gz'))
        ssim_list.append(metrics[0])
        psnr_list.append(metrics[1])
 
    ssim_list = np.array(ssim_list)
    psnr_list = np.array(psnr_list)
    print('MyASL Average SSIM:', np.round(np.mean(ssim_list),4),'±',np.round(np.std(ssim_list),4))
    print('MyASL Average PSNR:', np.round(np.mean(psnr_list),4),'±',np.round(np.std(psnr_list),4))

def prediction_pixel_guide_MyASL(model_name = 'train_PixelFM_HighnoiseGMASL_ch32v8attn8_earlystop_7500',
                        save_file_name ='MyASL_GuidedT1w',
                        fm_num_steps = 20,
                        file_name_prefix = 'MyASL_HighNoise_recon_20steps',
                        myasl_sr = 'MyASL'):
    #MyASL
    file_path = '/home/yunzhixu/Data/LDM_Data/'+myasl_sr+'.hdf5'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    hdf5data = h5py.File(file_path, 'r', libver='latest', swmr=True)
    ASL_data = hdf5data['ASL_LR'][()]
    ASL_HR  = hdf5data['ASL_HR'][()]
    T1_data = hdf5data['T1'][()]
    mask_data = T1_data.copy()
    mask_data[mask_data < 1] = 0
    mask_data[mask_data >= 1] = 1
    ASL_data = normalize_mri_array(ASL_data)
    ASL_HR = normalize_mri_array(ASL_HR)
    T1_data = normalize_mri_array(T1_data)  
    val_data = ASL_data
    target_data = ASL_HR


    config = load_yaml('/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/config.yml')
    config['model_config']['UNet_checkpoint'] = '/home/yunzhixu/Project/hydro_jupiter/'+str(model_name)+'/checkpoint.pth'
    save_dir = '/home/yunzhixu/Project/hydro_jupiter/' + model_name +'/'+ save_file_name
    os.makedirs(save_dir, exist_ok=True)

    pixel_infer = Pixel_Inference_Guided(config, device=device)
    ssim_list = []
    psnr_list = []
    for i in range(len(val_data)):
        print(f"Running guided inference on {len(val_data)} samples...")
        time0 = time.time()
        recon,metrics = pixel_infer.predict_guided_3d(T1_data[i], val_data[i], patch_size=[96, 96, 96], target=target_data[i],mask=mask_data[i],num_steps=fm_num_steps)
        save_nifit(recon, os.path.join(save_dir, f'{file_name_prefix}_{i}.nii.gz'))
        ssim_list.append(metrics[0])
        psnr_list.append(metrics[1])




if __name__ == '__main__':
    setup_seed(42)
    
    # 1. Train
    # train_GMASL_pixel()
    # prediction_pixel_guide_MyASL()
    # 2. Predict on Validation
    # predict_GMASL_pixel()

    # evaluate_test()
    
    # 3. Predict on Real Data
    predict_MyASL_pixel()