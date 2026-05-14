import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.modules.diffusionmodules import Encoder3D, Decoder3D
from tool.data_tool import Patch_data_multi
from monai.inferers import sliding_window_inference
from ldm.modules.diffusionmodules import UNetModel

from tool.model.diffusion_model.SRDiff import SR3DiffusionTrainer, SR3DiffusionSampler
from tool.model.diffusion_model.Diffusion import GaussianDiffusionSampler,GaussianDiffusionTrainer
from ldm.modules.diffusionmodules import Encoder3D,Decoder3D,VectorQuantizer3D
from tool.model import DDIMSampler,EDSR3D,DDPM
from tool.image_process import save_nifit, normalize_mri
from matplotlib import pyplot as plt
from tool.plot_tool import show_train_curve, plot_all_scalars
from tool.module.vae_module import VAE_Inference,AE_model
from tool.model.fusion_block import GatedFusionUnit,CrossAttentionFusionBlock,LinearCrossAttentionFusion,AdvancedGatedFusionUnit
from tool.image_process import save_nifit,crop_pad3D,normalize_mri,normalize_mri_array
import yaml
import glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tool.plot_tool import display_center_slices
from tool import SSIM3D,PSNR,MutualInformationLoss,GSloss_3D,SSIM
from tool.image_process import z_score_syn_tensor
from tool.model.diffusion_model import UNet
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

    
setup_seed(42)
def evaluate_metric(original, reconstructed, normalize=True,mask=None):
    ssimloss = SSIM3D(window_size=11)
    mil = MutualInformationLoss(nbins=128)
    psnr = PSNR()
    if mask is not None:
        original = original * mask
        reconstructed = reconstructed * mask
    if normalize:
        original, reconstructed = z_score_syn_tensor(original, reconstructed, windows=[-1, 6])
    original = torch.nan_to_num(original)
    reconstructed = torch.nan_to_num(reconstructed)
    
    psnr_val = psnr(original, reconstructed).item()
    ssim_val = ssimloss(original, reconstructed).item()
    return [ssim_val, psnr_val]

##checkpoint去除module:
def remove_module_from_state_dict(checkpoint):
    new_state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Remove "module." prefix
        else:
            new_state_dict[key] = value
        checkpoint = new_state_dict
    return checkpoint


class AE_Diffusion_T1w(nn.Module):
    """
    将输入噪声替换为T1w的正向扩散结果，兼容KL和VQ模式
    """
    def __init__(self,  
                    encoder,
                    decoder,
                    quant_conv,
                    post_quant_conv,
                    sampler,
                    mode='kl',
                    quantize=None,scale=0.3):
        super(AE_Diffusion_T1w, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.sampler = sampler
        self.mode = mode
        self.quantize = quantize
        self.scale = scale  # Scale factor for latent representation

    def forward(self, input):
        T1w = input[:, 0:1, ...]
        context = input[:, 1:2, ...]
        
        # Encode context
        y = self.encoder(context)
        z = self.quant_conv(y)
        if self.mode == 'kl':
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
        elif self.mode == 'vq' and self.quantize is not None:
            quant, _, _ = self.quantize(z)
            z = quant
        
        # Encode T1w
        T1w_y = self.quant_conv(self.encoder(T1w))

        if self.mode == 'kl':
            posterior_t1w = DiagonalGaussianDistribution(T1w_y)
            noise = posterior_t1w.sample()
        elif self.mode == 'vq' and self.quantize is not None:
            quant_t1w, _, _ = self.quantize(T1w_y)
            noise = quant_t1w
        
        # Diffusion process
        z = z * self.scale  # Scale the latent representation
        z_diffusion = self.sampler(noise, z)
        y = self.post_quant_conv(z_diffusion)
        reconstructions = self.decoder(y)
        return reconstructions


class AE_Diffusion_Cond(nn.Module):
    def __init__(self,  
                    encoder,
                    decoder,
                    quant_conv,
                    post_quant_conv,
                    sampler,
                    mode='kl',
                    quantize=None,
                    scale=0.3,
                    start_noise = 'guassian',
                    double_cond = 'concat',
                    fusion_block=None):
        '''
        start_noise: 'guassian' or 'prior'，使用先验图像还是高斯噪声作为扩散的起始变量
        double_cond: 'concat' or 'add_mean','add_mean_sqrt'，使用2个变量融合的方法，concat是通道拼接，add_mean是相加平均/2,'add_mean_sqrt'是相加开方平均/√2
        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.sampler = sampler
        self.mode = mode
        self.quantize = quantize
        self.scale = scale
        self.start_noise = start_noise
        self.double_cond = double_cond

        ##self.fusion_block
        self.fusion_block = fusion_block
    def forward(self, input):
        T1w = input[:, 0:1, ...] # This is not used for noise, but kept for signature consistency
        context = input[:, 1:, ...]
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)

        z_cond_list = []
        for i in range(context.shape[1]):
            cond_i = context[:, i:i+1, ...]
            y = self.encoder(cond_i)
            z = self.quant_conv(y)
            if self.mode == 'kl':
                posterior = DiagonalGaussianDistribution(z)
                z_sample = posterior.sample()
            elif self.mode == 'vq' and self.quantize is not None:
                z_sample, _, _ = self.quantize(z)
            z_cond_list.append(z_sample)
        
        # Combine conditional latents
        if len(z_cond_list) == 1:
            z_cond = z_cond_list[0]
        else:
            if self.double_cond == 'concat':
                z_cond = torch.cat(z_cond_list, dim=1)
            elif self.double_cond == 'add_mean':
                z_cond = (z_cond_list[0] + z_cond_list[1]) / 2
            elif self.double_cond == 'add_mean_sqrt':
                z_cond = (z_cond_list[0] + z_cond_list[1]) / (2**0.5)
            elif self.double_cond == 'gate':
                z_cond = self.fusion_block(z_cond_list[0], z_cond_list[1])
            elif self.double_cond == 'cross_attention':
                z_cond = self.fusion_block(z_cond_list[0], z_cond_list[1])
            else:
                raise ValueError('double_cond must be concat, add_mean, or add_mean_sqrt')
        
        # Determine starting noise for the ODE solve
        if self.start_noise == 'prior':
            strat_var_y = self.encoder(T1w)
            strat_var_z = self.quant_conv(strat_var_y)
            if self.mode == 'kl':
                noise = DiagonalGaussianDistribution(strat_var_z).sample()
            else: # vq
                noise, _, _ = self.quantize(strat_var_z)
        else: # 'guassian'
            noise = torch.randn_like(z_cond_list[0]) # Noise has shape of a single latent
        
        # Flow Matching sampling process
        # z_cond_scaled = z_cond * self.scale

        ##将单纯的scale改为标准化：
        z_mean= torch.mean(z_cond,dim=[0,2,3,4],keepdim=True)
        z_std= torch.std(z_cond,dim=[0,2,3,4],keepdim=True)
        z_cond_scaled = (z_cond - z_mean)/(z_std + 1e-6)


        # print('mean','std',torch.mean(z_cond_scaled),torch.std(z_cond_scaled))
        z_cond_scaled = torch.nan_to_num(z_cond_scaled, nan=0.0)

        

        
        # The sampler now takes noise and condition, and solves the ODE
        z_flow = self.sampler(noise, z_cond_scaled)
        z_flow = torch.nan_to_num(z_flow, nan=0.0)
        # z_flow =z_flow/self.scale
        ##de-standardize
        z_flow = z_flow * z_std + z_mean
        # print('mean','std',torch.mean(z_flow),torch.std(z_flow))
        
        y = self.post_quant_conv(z_flow)
        reconstructions = self.decoder(y)
        return reconstructions



class AE_UNet(nn.Module):
    """Wrapper for UNet model in latent space"""
    def __init__(self, encoder, decoder, quant_conv, post_quant_conv, model, mode='kl', quantize=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.model = model
        self.mode = mode
        self.quantize = quantize

    def forward(self, x):
        y = self.encoder(x)
        z = self.quant_conv(y)
        if self.mode == 'kl':
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
        elif self.mode == 'vq' and self.quantize is not None:
            quant, _, _ = self.quantize(z)
            z = quant
        z_diffusion = self.model(z)
        y = self.post_quant_conv(z_diffusion)
        return self.decoder(y)
    

class AE_UNet_Cond(nn.Module):
    '''
    用于使用单纯VAE的多变量融合预测 (This class does not use diffusion/flow and remains as is)
    '''
    def __init__(self,
                    encoder,
                    decoder,
                    quant_conv,
                    post_quant_conv,
                    mode='kl',
                    quantize=None,
                    scale=1,
                    double_cond = 'concat',
                    fusion_block=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.mode = mode
        self.quantize = quantize
        self.scale = scale
        self.double_cond = double_cond

        ##fusion_block
        self.fusion_block = fusion_block

    def forward(self, input):
        context = input[:, 1:, ...] # Input is assumed to be only condition(s)
        context = torch.nan_to_num(context, nan=0.0)
    
        z_cond_list = []
        for i in range(context.shape[1]):
            cond_i = context[:, i:i+1, ...]
            y = self.encoder(cond_i)
            z = self.quant_conv(y)
            if self.mode == 'kl':
                z_sample = DiagonalGaussianDistribution(z).sample()
            elif self.mode == 'vq' and self.quantize is not None:
                z_sample, _, _ = self.quantize(z)
            z_cond_list.append(z_sample)
            
        if len(z_cond_list) == 1:
            z_cond = z_cond_list[0]
        else:
            if self.double_cond == 'concat':
                z_cond = torch.cat(z_cond_list, dim=1)
            elif self.double_cond == 'add_mean':
                z_cond = (z_cond_list[0] + z_cond_list[1]) / 2
                print('z_cond',z_cond.shape)
                print('mean std',torch.mean(z_cond),torch.std(z_cond))
            elif self.double_cond == 'add_mean_sqrt':
                z_cond = (z_cond_list[0] + z_cond_list[1]) / (2**0.5)
            elif self.double_cond == 'gate':
                z_cond = self.fusion_block(z_cond_list[0], z_cond_list[1])
            elif self.double_cond == 'cross_attention':
                z_cond = self.fusion_block(z_cond_list[0], z_cond_list[1])

            else:
                raise ValueError('double_cond must be concat or add_mean or add_mean_sqrt')
                
        z_encod = z_cond #* self.scale
        y = self.post_quant_conv(z_encod)
        reconstructions = self.decoder(y)
        return reconstructions




class AE_Diffusion(nn.Module):
    """Wrapper for diffusion model in latent space"""
    def __init__(self, encoder, decoder, quant_conv, post_quant_conv, sampler, mode='kl', quantize=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quant_conv = quant_conv
        self.post_quant_conv = post_quant_conv
        self.sampler = sampler
        self.mode = mode
        self.quantize = quantize

    def forward(self, x):
        y = self.encoder(x)
        z = self.quant_conv(y)
        if self.mode == 'kl':
            posterior = DiagonalGaussianDistribution(z)
            z = posterior.sample()
        elif self.mode == 'vq' and self.quantize is not None:
            quant, _, _ = self.quantize(z)
            z = quant
        noise = torch.randn_like(z)
        z_diffusion = self.sampler(noise, z)
        y = self.post_quant_conv(z_diffusion)
        return self.decoder(y)

class DM_Cat_Module(nn.Module):
    """
    DM_Cat_Module: Diffusion Model Module based on latent diffusion with channel concatenation.
    This module implements a diffusion model that operates on latent representations of 3D medical images. 
    It supports both KL-based and vector quantization (VQ)-based variational autoencoders (VAEs) for encoding 
    and decoding the latent space. The diffusion model is conditioned on additional input channels, enabling 
    channel concatenation for guided generation.
    """
    """Diffusion Model Module based on latent diffusion with channel concatenation"""
    def __init__(self, config):
        '''
        
        '''
        super().__init__()
        self.device = device
        self.config = config
        self.global_step = 0
        self.vae_type = self.config['model_config']['Type']
        
        # Initialize components
        self._init_components(config)
        
        # Data loaders
        self.train_loader = self._get_data_loader(
            config['data_config']['train_paths'], 
            is_train=True,
            batch_size=config['data_config']['batch_size']
        )
        self.val_loader = self._get_data_loader(
            config['data_config']['val_paths'], 
            is_train=False,
            batch_size=config['data_config']['batch_size']
        )
        
        # Logging
        os.system('rm -rf ' + config['train_config']['log_dir'])
        os.makedirs(config['train_config']['log_dir'], exist_ok=True)
        self.writer = SummaryWriter(log_dir=config['train_config']['log_dir'])
        self.log_dir = config['train_config']['log_dir']
        
        # Save directory
        os.makedirs(config['train_config']['save_dir'], exist_ok=True)
        with open(os.path.join(config['train_config']['save_dir'], 'config.txt'), 'w') as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        # Save config to a YAML file
        config_path = os.path.join(config['train_config']['save_dir'], 'config.yml')
        with open(config_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)

    def _init_components(self, config):
        """Initialize model components"""
        # VAE components
        vae_config = config['model_config']['VAE']


        self.encoder = Encoder3D(
            in_channels=vae_config['in_ch'],
            out_ch=vae_config['out_ch'],
            ch = vae_config['channels'],
            z_channels=vae_config['z_channels'],
            ch_mult=vae_config['ch_mult'],
            num_res_blocks=vae_config['num_res_blocks'],
            attn_resolutions=vae_config['attn_resolutions'],
            dropout=vae_config['dropout'],
            double_z=vae_config['double_z'],
            middle_block=vae_config['middle_block'],
            resolution=self.config['data_config']['patch_size'][0],
            resamp_with_conv=True,
             ).to(device)

        self.decoder = Decoder3D(
            in_channels=vae_config['in_ch'],
            out_ch=vae_config['out_ch'],
            ch = vae_config['channels'],
            z_channels=vae_config['z_channels'],
            ch_mult=vae_config['ch_mult'],
            num_res_blocks=vae_config['num_res_blocks'],
            attn_resolutions=vae_config['attn_resolutions'],
            dropout=vae_config['dropout'],
            double_z=vae_config['double_z'],
            middle_block=vae_config['middle_block'],
            resolution=self.config['data_config']['patch_size'][0],
            resamp_with_conv=True,
        ).to(device)
        
        if self.vae_type == 'kl':
            embed_dim = vae_config['z_channels'] 
            self.quant_conv = nn    .Conv3d(2*embed_dim, 2*embed_dim, 1).to(device)
        else:  # vq
            embed_dim = vae_config['embed_dim']
            self.quant_conv = nn.Conv3d(embed_dim, embed_dim, 1).to(device)
            self.quantize = VectorQuantizer3D(
                self.config['model_config']['VAE']['n_embed'],
                embed_dim,
                beta=0.25,
                remap=None,
                sane_index_shape=False
            ).to(device)
        self.post_quant_conv = nn.Conv3d(embed_dim, embed_dim, 1).to(device)


        ##查看是否双GPU：



        ##加载权重
        mode = self.vae_type
        load_vae_fun = config['model_config']['load_vae_fun']
        if config['model_config']['vae_checkpoint'] is not None:
            model_dir = config['model_config']['vae_checkpoint']
            if load_vae_fun == 'model':
                encoder = torch.load(os.path.join(model_dir, 'encoder.pth')).to(self.device)
                decoder = torch.load(os.path.join(model_dir, 'decoder.pth')).to(self.device)
                quant_conv = torch.load(os.path.join(model_dir, 'quant_conv.pth')).to(self.device)
                post_quant_conv = torch.load(os.path.join(model_dir, 'post_quant_conv.pth')).to(self.device)

                self.encoder.load_state_dict(remove_module_from_state_dict(encoder.state_dict()))
                self.decoder.load_state_dict(remove_module_from_state_dict(decoder.state_dict()))
                self.quant_conv.load_state_dict(remove_module_from_state_dict(quant_conv.state_dict()))
                self.post_quant_conv.load_state_dict(remove_module_from_state_dict(post_quant_conv.state_dict()))
                                             



                print('state_dict load vae model',self.encoder.state_dict().keys()) 
                if mode == 'vq':
                    self.quantize = torch.load(os.path.join(model_dir, 'quantize.pth')).to(self.device)
                else:
                    self.quantize = None
            elif load_vae_fun == 'state_dict':
                checkpoint = torch.load(model_dir, map_location=self.device)
                self.encoder.load_state_dict(remove_module_from_state_dict(checkpoint['encoder']))
                self.decoder.load_state_dict(remove_module_from_state_dict(checkpoint['decoder']))
                self.quant_conv.load_state_dict(remove_module_from_state_dict(checkpoint['quant_conv']))
                self.post_quant_conv.load_state_dict(remove_module_from_state_dict(checkpoint['post_quant_conv']))
                if mode == 'vq':
                    self.quantize.load_state_dict(remove_module_from_state_dict(checkpoint['quantize']))
        

        #Eval模型
  
        self.encoder.eval()
        self.decoder.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        if self.vae_type == 'vq':
            # self.quantize = nn.DataParallel(self.quantize,device_ids=[0,1])
            self.quantize.eval()

        
        # Diffusion model
        unet_params = config['model_config']['UNet']
        self.model = UNetModel(**unet_params
        ).to(device)
        # Load UNet weights if available
        if config['model_config']['UNet_checkpoint'] is not None:
            unet_checkpoint = torch.load(config['model_config']['UNet_checkpoint'], map_location=self.device)
            if 'model' in unet_checkpoint:
                self.model.load_state_dict(remove_module_from_state_dict(unet_checkpoint['model']))
            else:
                self.model.load_state_dict(unet_checkpoint)
        
 
    
        
        # Parallelize if multiple GPUs
        ##判断权重是否有module
        if any(key.startswith("module.") for key in self.encoder.state_dict().keys()):
            print("State dict contains 'module.' prefix.")
        else:
            if torch.cuda.device_count() > 1 :
                print(f"Using {torch.cuda.device_count()} GPUs for training.")
                self.encoder = nn.DataParallel(self.encoder,device_ids=[0,1])
                self.decoder = nn.DataParallel(self.decoder,device_ids=[0,1])
                self.quant_conv = nn.DataParallel(self.quant_conv,device_ids=[0,1])
                self.post_quant_conv = nn.DataParallel(self.post_quant_conv,device_ids=[0,1])
                if self.vae_type == 'vq':
                    self.quantize = nn.DataParallel(self.quantize,device_ids=[0,1])
                self.model = nn.DataParallel(self.model,device_ids=[0,1])
        # Trainer and optimizer
        self.trainer = SR3DiffusionTrainer(self.model, 1e-4, 0.02, 1000)
        self.sampler = SR3DiffusionSampler(self.model, 1e-4, 0.02,1000)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['train_config']['lr'],
            betas=(0.9, 0.95),
            weight_decay=1e-4)
        self.scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=config['train_config']['lr_scheduler'],
            threshold = 1e-6,
            threshold_mode = 'rel',
            cooldown=0,
            min_lr = 0,
            eps = 1e-08) 


    def _get_data_loader(self, data_paths, is_train, batch_size):
        """Create data loader"""
        dataset = Patch_data_multi(
            data_path=data_paths,
            data_name=self.config['data_config']['get_data_name'],
            patch_size=self.config['data_config']['patch_size'],
            all_data=self.config['data_config']['all_data'],
            trainable=is_train,
            augmentation=is_train,
            data_prenorm='z_score_norm',
            patch_training=True,
            is_hot_label=self.config['data_config']['hot_label'],
            channel_first=False,
            patch_pro = self.config['data_config']['patch_pro'],
            # mask_name=['seg'],
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

    def train_step(self, batch):
        """Single training step"""
        image = batch[0].permute(0,4,1,2,3).to(device)
        cond = batch[1].permute(0,4,1,2,3).to(device)

        
        # Encode to latent space

        with torch.no_grad():
            z_image = self.encoder(image)
            z_image = self.quant_conv(z_image)
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_image)
                z_image = posterior.sample()
            else:  # vq
                z_image, _, _ = self.quantize(z_image)
            
            z_cond = self.quant_conv(self.encoder(cond))
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_cond)
                z_cond = posterior.sample()
            else:  # vq
                z_cond, _, _ = self.quantize(z_cond)
        
        # Train diffusion model
        ##scale z
        z_image = z_image * self.config['model_config']['z_scale']
        z_cond = z_cond * self.config['model_config']['z_scale']
        loss = self.trainer(z_image, z_cond).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ##加学习率衰减：
        self.scheduler.step(loss)
        
        # self.writer.add_scalar('loss', loss.item(), self.global_step)
        self.global_step += 1
        
        return loss.item()

    def validate(self):
        """Run validation"""
        self.eval()
        val_losses = []
        val_item = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                image = batch[0].permute(0,4,1,2,3).to(device)
                cond = batch[1].permute(0,4,1,2,3).to(device)
                
                # Encode to latent space
                z_image = self.quant_conv(self.encoder(image))
                if self.vae_type == 'kl':
                    posterior = DiagonalGaussianDistribution(z_image)
                    z_image = posterior.sample()
                else:  # vq
                    z_image, _, _ = self.quantize(z_image)
                
                z_cond = self.quant_conv(self.encoder(cond))
                if self.vae_type == 'kl':
                    posterior = DiagonalGaussianDistribution(z_cond)
                    z_cond = posterior.sample()
                else:  # vq
                    z_cond, _, _ = self.quantize(z_cond)

                # print('z_image shape:', z_image.shape, 'z_cond shape:', z_cond.shape)
                z_image = z_image * self.config['model_config']['z_scale']
                z_cond = z_cond * self.config['model_config']['z_scale']
                loss = self.trainer(z_image, z_cond).mean()
                val_losses.append(loss.item())
                val_item += 1
                if val_item % 5 == 0:
                    print(f"Validation step {val_item}, loss: {loss.item():.4f}")
                    break


                ##还原

        y = self.inference(cond)
        # Save validation images
        save_nifit(y[0, 0, :, :, :].detach().cpu().numpy(), 
                    os.path.join(self.config['train_config']['save_dir'], f'val_sample_{self.global_step}.nii.gz'))
        recon = y[0, 0, :, :, :].detach().cpu().numpy()
        plt.figure(figsize=(15, 5),dpi=100)
        plt.subplot(1, 3, 1)
        plt.imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')
        plt.title('Axial Slice')
        plt.subplot(1, 3, 2)
        plt.imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
        plt.title('Coronal Slice')
        plt.subplot(1, 3, 3)
        plt.imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
        plt.title('Sagittal Slice')
        plt.imshow(recon[::,recon.shape[2]//2], cmap='gray')
        plt.savefig(os.path.join(self.config['train_config']['save_dir'], f'val_sample_{self.global_step}.png'))
                
        
        self.train()
        return np.mean(val_losses)

    def inference(self, cond):
        """Generate samples from conditioning"""
        self.eval()
        with torch.no_grad():
            # Encode conditioning
            z_cond = self.quant_conv(self.encoder(cond))
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_cond)
                z_cond = posterior.sample()
            elif self.vae_type == 'vq' and self.quantize is not None:
                quant, _, _ = self.quantize(z_cond)
                z_cond = quant
                        # Create sampler
            sampler = SR3DiffusionSampler(self.model, 1e-4, 0.02, 1000)
            # sampler = DDIMSampler(device,n_steps=1000,ddim_step=100,net=self.model,eta=0,infer_T=None,condition_fun='SR3')
            
            # Sample noise
            noise = torch.randn_like(z_cond)
            print('shape:',noise.shape,z_cond.shape)
            z_cond = z_cond * self.config['model_config']['z_scale']
            # Generate sample in latent space
            z_sample = sampler(noise, z_cond)
        
            
            # Decode to image space
            y = self.post_quant_conv(z_sample)
        return self.decoder(y)
    
    def predict_full_image(self, full_image,cond_image, patch_size, model,sample_fun='SR3'):
        """
        Perform inference on a full image using AE_UNet with sliding window.

        Args:
            full_image (torch.Tensor): The full image tensor of shape (C, D, H, W).
            patch_size (tuple): The size of the patch (D, H, W).
            overlap (float): The overlap ratio between patches (0.0 to 1.0).

        Returns:
            torch.Tensor: The reconstructed full image.
        """
        full_image = torch.tensor(full_image, dtype=torch.float32)
        cond_image = torch.tensor(cond_image, dtype=torch.float32)
        self.eval()

        if  sample_fun == 'Guass':
            sampler = GaussianDiffusionSampler(model, 1e-4, 0.02,1000,infer_T=None)
        elif sample_fun == 'SR3':
            sampler = SR3DiffusionSampler(model, 1e-4, 0.02,1000)
        elif sample_fun == 'DDIM':
            sampler = DDIMSampler(device,n_steps=1000,ddim_step=100,net=model,eta=0,infer_T=None)##
        elif sample_fun == 'SR3+DDIM':
            sampler = DDIMSampler(device,n_steps=1000,ddim_step=100,net=model,eta=0,infer_T=None,condition_fun='SR3')##
        else:
            raise ValueError('sample_fun must be Guass or SR3 or DDIM')
        

        AE_DM_Model = AE_Diffusion_T1w(self.encoder,self.decoder,self.quant_conv,self.post_quant_conv,sampler)
        with torch.no_grad():
            # Ensure the input is on the correct device
            full_image = full_image.to(device)
            test_data = full_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            cond_data = cond_image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            test_data = torch.cat((test_data, cond_data), dim=1)  # Concatenate along channel dimension

            # Perform sliding window inference
            reconstructed_image = sliding_window_inference(test_data,self.config['data_config']['patch_size'],2,AE_DM_Model)

        return reconstructed_image

    def train_loop(self, total_steps):
        """Full training loop"""
        epoch_iterator = tqdm(
            self.train_loader,
            desc=f"Training (loss=X.X)",
            dynamic_ncols=True
        )
        
        for batch in epoch_iterator:
            if self.global_step >= total_steps:
                break
                
            loss = self.train_step(batch)
            epoch_iterator.set_description(f"Training (loss={loss:.4f})")
            epoch_iterator.update(1)
            self.writer.add_scalar('train/loss', loss, self.global_step)
            self.global_step += 1
            ##lr 
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Validation and checkpointing
            if self.global_step % self.config['train_config']['save_interval'] == 0:
                val_loss = self.validate()
                self.save_checkpoint()
                self.writer.add_scalar('val/loss', val_loss, self.global_step)
                if self.config['train_config']['save_interval']>100:
                    plot_all_scalars(self.log_dir, self.config['train_config']['save_dir']+'/all_curve.png',step_num=1)
                # self.predict_full_image(
                
    def save_checkpoint(self):
        """Save model checkpoint"""
        torch.save({
        'encoder': self.encoder.state_dict(),
        'decoder': self.decoder.state_dict(),
        'quant_conv': self.quant_conv.state_dict(),
        'post_quant_conv': self.post_quant_conv.state_dict(),
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'quantize': self.quantize.state_dict() if self.vae_type == 'vq' else None
        }, os.path.join(self.config['train_config']['save_dir'], 'checkpoint.pth'))

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.quant_conv.load_state_dict(checkpoint['quant_conv'])
        self.post_quant_conv.load_state_dict(checkpoint['post_quant_conv'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

def get_default_config():
    """Return default configuration for DMModule"""
    train_name = 'VQ_dm_cat_train'
    return {
        'model_config': {
            'Type': 'vq',
            'VAE': {
                "in_ch": 1,
                "z_channels": 16,
                "embed_dim": 16,
                "n_embed": 1024,
                "channels": 32,
                "ch_mult": (1, 2, 4, 8),
                "num_res_blocks": 2,
                "attn_resolutions": [],
                "out_ch": 1,
                "dropout": 0.0,
                "middle_block": False,
                "double_z": False,
            },
            'vae_checkpoint': './checkpoints/vae.pth',
            'UNet': {
                "dims": 3,
                "image_size": 12,
                "in_channels": 32,  # Assuming 2 * embed_dim where embed_dim = 16
                "out_channels": 16,  # Matching embed_dim
                "model_channels": 64,
                "attention_resolutions": [4],
                "num_res_blocks": 1,
                "channel_mult": [1, 2, 4],
                "num_head_channels": 8,
                "context_dim": None,
                "use_spatial_transformer": False,},
            'UNet_checkpoint': None,
            'z_scale': 0.3,#用于对vae的latent进行缩放
        },
        'train_config': {
            'lr': 1e-4,
            'lr_scheduler': 1000,
            'log_dir': '/data/yunzhixu/medical_fusion/hydro_source/source/run/'+train_name,
            'train_name': train_name,
            'root_path': '/data/yunzhixu/medical_fusion/hydro_source/',
            'save_dir': '/data/yunzhixu/medical_fusion/hydro_source/'+train_name,
            'save_interval': 1000,
            'val_interval': 10,
            'val_image': None,
            'val_target': None,
            'label_dict': None,
        },
        'data_config': {
            'train_paths': ['./data/train'],
            'val_paths': ['./data/val'],
            'get_data_name': ['img','img'],
            'batch_size': 4,
            'patch_size': [112, 112, 112],
            'hot_label': False,
        }
    }




class SVR_DMModule(DM_Cat_Module):
    '''
    用于实现SVR重建的扩散模型

    '''
    def __init__(self, config, double_cond='concat'):
        super().__init__(config)
        self.double_cond = double_cond

        ##根据double_cond选择fusion_block
        if self.double_cond == 'gate':
            self.fusion_block = AdvancedGatedFusionUnit(in_channels=self.config['model_config']['VAE']['z_channels'],
                                                kernel_size=3).to(self.device)
            ##给fusion block添加上优化器
            self.optimizer = torch.optim.AdamW(list(self.model.parameters())+list(self.fusion_block.parameters()), lr=self.config['train_config']['lr'], betas=(0.9, 0.95), weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=config['train_config']['lr_scheduler'])
        elif self.double_cond == 'cross_attention':
            # self.fusion_block = CrossAttentionFusionBlock(in_channels=self.config['model_config']['VAE']['z_channels'],
            #                                               nuxsm_heads=4,
            #                                               head_dim=32).to(self.device)
            #LinearCrossAttentionFusion
            self.fusion_block = LinearCrossAttentionFusion(in_channels=self.config['model_config']['VAE']['z_channels'],num_heads=4, head_dim=32)
            self.optimizer = torch.optim.AdamW(list(self.model.parameters())+list(self.fusion_block.parameters()), lr=self.config['train_config']['lr'], betas=(0.9, 0.95), weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=config['train_config']['lr_scheduler'])



    def train_step(self, batch):
        """Modified single training step"""
        image = batch[0].permute(0, 4, 1, 2, 3).to(self.device)
        cond0 = batch[1].permute(0, 4, 1, 2, 3).to(self.device)
        cond1 = batch[2].permute(0, 4, 1, 2, 3).to(self.device)

        # Encode to latent space
        with torch.no_grad():
            z_image = self.encoder(image)
            z_image = self.quant_conv(z_image)
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_image)
                z_image = posterior.sample()
            else:  # vq
                z_image, _, _ = self.quantize(z_image)

            z_cond0 = self.quant_conv(self.encoder(cond0))
            z_cond1 = self.quant_conv(self.encoder(cond1))
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_cond0)
                z_cond0 = posterior.sample()
                posterior = DiagonalGaussianDistribution(z_cond1)
                z_cond1 = posterior.sample()
            else:  # vq
                z_cond0, _, _ = self.quantize(z_cond0)
                z_cond1, _, _ = self.quantize(z_cond1)

            ##normalize z_cond0 and z_cond1,z_image
        z_cond0_mean, z_cond0_std= torch.mean(z_cond0,dim=[0,2,3,4],keepdim=True),torch.std(z_cond0,dim=[0,2,3,4],keepdim=True)
        z_cond1_mean, z_cond1_std= torch.mean(z_cond1,dim=[0,2,3,4],keepdim=True),torch.std(z_cond1,dim=[0,2,3,4],keepdim=True)
        z_image_mean, z_image_std= torch.mean(z_image,dim=[0,2,3,4],keepdim=True),torch.std(z_image,dim=[0,2,3,4],keepdim=True)
        z_cond0 = (z_cond0 - z_cond0_mean)/(z_cond0_std + 1e-6)
        z_cond1 = (z_cond1 - z_cond1_mean)/(z_cond1_std + 1e-6)
        z_image = (z_image - z_image_mean)/(z_image_std + 1e-6)
        z_cond0 = torch.nan_to_num(z_cond0, nan=0.0)
        z_cond1 = torch.nan_to_num(z_cond1, nan=0.0)
        z_image = torch.nan_to_num(z_image, nan=0.0)

        if self.double_cond == 'concat':
            z_cond = torch.cat((z_cond0, z_cond1), dim=1)  # Concatenate along channel dimension
        elif self.double_cond == 'add_mean':
            z_cond = (z_cond0 + z_cond1) / 2
        elif self.double_cond == 'add_mean_sqrt':
            z_cond = (z_cond0 + z_cond1) / (2**0.5)
        elif self.double_cond == 'gate' or self.double_cond == 'cross_attention':
            z_cond = self.fusion_block(z_cond0, z_cond1)
        else:
            raise ValueError("double_cond must be 'concat', 'add_mean', or 'add_mean_sqrt','gate','cross_attention'")

        
        
        
        
        # Train diffusion model
        # z_image = z_image * self.config['model_config']['z_scale']
        # z_cond = z_cond * self.config['model_config']['z_scale']
        loss = self.trainer(z_image, z_cond).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        ##学习率衰减：
        self.scheduler.step(loss)

        self.global_step += 1

        return loss.item()

    def validate(self,val_step=1):
        """Modified validation"""
        self.eval()
        val_losses = []
        val_item = 0

        with torch.no_grad():
            for batch in self.val_loader:
                image = batch[0].permute(0, 4, 1, 2, 3).to(self.device)
                cond0 = batch[1].permute(0, 4, 1, 2, 3).to(self.device)
                cond1 = batch[2].permute(0, 4, 1, 2, 3).to(self.device)
                save_nifit(image[0, 0, :, :, :].detach().cpu().numpy(), 
                            os.path.join(self.config['train_config']['save_dir'], f'val_image_{self.global_step}.nii.gz'))

                # Encode to latent space
                z_image = self.encoder(image)
                z_image = self.quant_conv(z_image)
                if self.vae_type == 'kl':
                    print('kl')
                    posterior = DiagonalGaussianDistribution(z_image)
                    z_image = posterior.sample()
                else:  # vq
                    print('vq')
                    z_image, _, _ = self.quantize(z_image)

                z_cond0 = self.quant_conv(self.encoder(cond0))
                z_cond1 = self.quant_conv(self.encoder(cond1))
                if self.vae_type == 'kl':
                    posterior = DiagonalGaussianDistribution(z_cond0)
                    z_cond0 = posterior.sample()
                    posterior = DiagonalGaussianDistribution(z_cond1)
                    z_cond1 = posterior.sample()
                else:  # vq
                    z_cond0, _, _ = self.quantize(z_cond0)
                    z_cond1, _, _ = self.quantize(z_cond1)
                z_cond0_mean, z_cond0_std= torch.mean(z_cond0,dim=[0,2,3,4],keepdim=True),torch.std(z_cond0,dim=[0,2,3,4],keepdim=True)
                z_cond1_mean, z_cond1_std= torch.mean(z_cond1,dim=[0,2,3,4],keepdim=True),torch.std(z_cond1,dim=[0,2,3,4],keepdim=True)
                z_image_mean, z_image_std= torch.mean(z_image,dim=[0,2,3,4],keepdim=True),torch.std(z_image,dim=[0,2,3,4],keepdim=True)
                z_cond0 = (z_cond0 - z_cond0_mean)/(z_cond0_std + 1e-6)
                z_cond1 = (z_cond1 - z_cond1_mean)/(z_cond1_std + 1e-6)
                z_image = (z_image - z_image_mean)/(z_image_std + 1e-6)
    
                if self.double_cond == 'concat':
                    z_cond = torch.cat((z_cond0, z_cond1), dim=1)
                elif self.double_cond == 'add_mean':
                    z_cond = (z_cond0 + z_cond1) / 2
                elif self.double_cond == 'add_mean_sqrt':
                    z_cond = (z_cond0 + z_cond1) / (2**0.5)
                elif self.double_cond == 'gate' or self.double_cond == 'cross_attention':
                    z_cond = self.fusion_block(z_cond0, z_cond1)
                else:
                    raise ValueError("double_cond must be 'concat', 'add_mean', or 'add_mean_sqrt'")


                # z_image_s = z_image * self.config['model_config']['z_scale']
                # z_cond_s = z_cond * self.config['model_config']['z_scale']
                z_image_s = z_image
                z_cond_s = z_cond
 
                loss = self.trainer(z_image_s, z_cond_s).mean()
                val_losses.append(loss.item())
                val_item += 1
                if val_item % val_step == 0:
                    print(f"Validation step {val_item}, loss: {loss.item():.4f}")
                    break
        # y = self.inference(cond0[0:1], cond1[0:1])
        # # Save validation images
        # save_nifit(y[0, 0, :, :, :].detach().cpu().numpy(), 
        #             os.path.join(self.config['train_config']['save_dir'], f'val_sample_{self.global_step}.nii.gz'))
        # recon = y[0, 0, :, :, :].detach().cpu().numpy()
        # plt.figure(figsize=(15, 5),dpi=100)
        # plt.subplot(1, 3, 1)
        # plt.imshow(recon[:, :, recon.shape[2] // 2], cmap='gray')
        # plt.title('Axial Slice')
        # plt.subplot(1, 3, 2)
        # plt.imshow(recon[:, recon.shape[1] // 2, :], cmap='gray')
        # plt.title('Coronal Slice')
        # plt.subplot(1, 3, 3)
        # plt.imshow(recon[recon.shape[0] // 2, :, :], cmap='gray')
        # plt.title('Sagittal Slice')
        # plt.imshow(recon[::,recon.shape[2]//2], cmap='gray')
        # plt.savefig(os.path.join(self.config['train_config']['save_dir'], f'val_sample_{self.global_step}.png'))


        val_data = []
        cond_data = []
        hdf5_path = self.config['data_config']['val_paths'][0]
        print('hdf5_path',hdf5_path)
        hdf5 = glob.glob(hdf5_path+'*.hdf5')

        for i in [1]:
            hdf5data = h5py.File(hdf5[i], 'r', libver='latest', swmr=True)
            img = hdf5data['img'][()]
            y_img= hdf5data['resampled_y'][()]
            z_img = hdf5data['resampled_z'][()]
            img = normalize_mri_array(img)
            y_img = normalize_mri_array(y_img)
            z_img = normalize_mri_array(z_img)
            img_shape= img.shape
            y_img= crop_pad3D(y_img, img_shape)
            z_img= crop_pad3D(z_img, img_shape)
            y_img = np.expand_dims(y_img, axis=0)
            z_img = np.expand_dims(z_img, axis=0)
            print('img shape:', img.shape,y_img.shape, z_img.shape)
            condition = np.concatenate([y_img, z_img], axis=0)
            cond_data.append(condition)
            # cond_data = z_img[np.newaxis,:,:,:]
            # print('lr_data',lr_data.shape)
            val_data.append(img)
        val_data = np.array(val_data)[0]
        cond_data = np.array(cond_data)[0]
    

        cond0_tensor = torch.from_numpy(cond_data[0]).float().unsqueeze(0).unsqueeze(0)
        cond1_tensor = torch.from_numpy(cond_data[1]).float().unsqueeze(0).unsqueeze(0)
        # Create a placeholder for the "T1w" input required by AE_Flow_Cond's signature
        # It won't be used for noise generation if start_noise='guassian'
        dummy_t1w = torch.zeros_like(cond0_tensor)
        # The input tensor for sliding window now includes the dummy T1w
        input_tensor = torch.cat((dummy_t1w, cond0_tensor, cond1_tensor), dim=1).to(self.device)
        ##load model
        self.sampler = SR3DiffusionSampler(self.model,1e-4, 0.02,1000)
        quantize =None
        self.predict_model = AE_Diffusion_Cond(
        self.encoder, self.decoder, self.quant_conv, self.post_quant_conv,
        self.sampler, self.config['model_config']['Type'], quantize, self.config['model_config']['z_scale'],
        start_noise='guassian', double_cond=self.double_cond,fusion_block=self.fusion_block
        )
        print('input_tensor',input_tensor.shape)
        with torch.no_grad():
            reconstruction = sliding_window_inference(
                input_tensor, self.config['data_config']['patch_size'], 1, self.predict_model, overlap=0.25, mode='gaussian')
        recon = reconstruction.cpu().numpy()[0, 0]
        save_dir = self.config['train_config']['save_dir']
        display_center_slices(recon, title="Reconstructed SVR Flow", save_path=os.path.join(save_dir, f'recon_{self.global_step}.png'))
        display_center_slices(val_data, title="Original Image", save_path=os.path.join(save_dir, 'original_img.png'))
                
        
        self.train()
        return np.mean(val_losses)

    def inference(self, cond0,cond1):
        """Modified inference"""
        self.eval()
        with torch.no_grad():
            # Encode conditioning
            z_cond0 = self.quant_conv(self.encoder(cond0))
            z_cond1 = self.quant_conv(self.encoder(cond1))
            if self.vae_type == 'kl':
                posterior = DiagonalGaussianDistribution(z_cond0)
                z_cond0 = posterior.sample()
                posterior = DiagonalGaussianDistribution(z_cond1)
                z_cond1 = posterior.sample()
            elif self.vae_type == 'vq' and self.quantize is not None:
                quant0, _, _ = self.quantize(z_cond0)
                quant1, _, _ = self.quantize(z_cond1)
                z_cond0 = quant0
                z_cond1 = quant1

            z_cond = torch.cat((z_cond0, z_cond1), dim=1)  # Concatenate along channel dimension

            # Create sampler
            sampler = SR3DiffusionSampler(self.model, 1e-4, 0.02, 1000)

            # Sample noise
            noise = torch.randn_like(z_cond0)
            z_cond = z_cond * self.config['model_config']['z_scale']

            # Generate sample in latent space
            z_sample = sampler(noise, z_cond)

            ##还原缩放
            z_sample = z_sample / self.config['model_config']['z_scale']

            # Decode to image space

            y = self.post_quant_conv(z_sample)
            y = self.decoder(y)
        return y


    def save_checkpoint(self):
        # ... (This method remains identical) ...
        torch.save({
        'encoder': self.encoder.state_dict(),
        'decoder': self.decoder.state_dict(),
        'quant_conv': self.quant_conv.state_dict(),
        'post_quant_conv': self.post_quant_conv.state_dict(),
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'quantize': self.quantize.state_dict() if self.vae_type == 'vq' else None,
        'fusion_block': self.fusion_block.state_dict() if self.double_cond in ['gate', 'cross_attention'] else None
        }, os.path.join(self.config['train_config']['save_dir'], 'checkpoint.pth'))


class LDM_Inference(VAE_Inference):
    """
    LDM_Inference: Inference class for Latent Diffusion Model (LDM) with VAE encoding and DM-based generation.
    """
    def __init__(self, config,dm_model_dir=None, vae_model_dir=None,mode='vq',load_fun='state_dict',patch_size=None,only_ae=False, device='cuda'):
        '''
        only_ae:推理时是否只使用ae模型，便于测试无扩散模型的重建效果
        '''
        if vae_model_dir is None:
            vae_model_dir = config['model_config']['vae_checkpoint']
        vae_model_config = config['model_config']['VAE']
        if patch_size is None:
            patch_size = config['data_config']['patch_size']
        self.device = device
        self.mode = mode
        self.only_ae = only_ae
    


        self._init_VAE(vae_model_dir, vae_model_config, load_fun, patch_size, mode)
        if not self.only_ae:
            print('self not only ae')
            self._init_UNet(config)
        else:
            self.cond_ldm = None
            self.sampler = None

        self.ae_model = AE_model(
            self.encoder, 
            self.decoder,
            self.quant_conv,
            self.post_quant_conv,
            self.quantize,
            mode = mode,
        ).to(self.device)
    
    def _init_UNet(self, config):
        """
        Initialize the UNet model for latent diffusion.
        
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        unet_params = config['model_config']['UNet']
        self.model = UNetModel(**unet_params).to(device)
        checkpoint = torch.load(config['model_config']['UNet_checkpoint'], map_location=device)
        self.model.load_state_dict(remove_module_from_state_dict(checkpoint['model']))
        self.model.eval()
        self.sampler = SR3DiffusionSampler(self.model, 1e-4, 0.02, 1000)
        # self.sampler = DDIMSampler(device,n_steps=1000,ddim_step=500,net=self.model,eta=0,infer_T=None,condition_fun='SR3')
        print('self.mode',self.mode)

        self.cond_ldm = AE_Diffusion_Cond(
            encoder=self.encoder,
                decoder=self.decoder,
                quant_conv=self.quant_conv,
                post_quant_conv=self.post_quant_conv,
                sampler=self.sampler,
                mode=self.mode,
                quantize=self.quantize,
                scale=config['model_config']['z_scale'],
                start_noise='guassian'
    
        )
        self.scale = config['model_config']['z_scale']

    def inference(self, context):
        
        """
        Perform inference using the Latent Diffusion Model (LDM).
        
        Args:
            cond (torch.Tensor): Conditioning input tensor.
        
        Returns:
            torch.Tensor: Generated output tensor.
        """
        # if context.shape[1] != 1:
        for i in range(context.shape[1]):
            locals()['context'+str(i)] = context[:, i:i+1, ...]
        self.encoder.eval()
        self.decoder.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        if self.mode == 'vq' and self.quantize is not None:
            self.quantize.eval()

        with torch.no_grad():
            for i in range(context.shape[1]):
                cond = locals()['context'+str(i)]
                y = self.encoder(cond)
                z = self.quant_conv(y)
                if self.mode == 'kl':
                    posterior = DiagonalGaussianDistribution(z)
                    z = posterior.sample()
                    locals()['z_cond'+str(i)] = z
                elif self.mode == 'vq' and self.quantize is not None:
                    print('vq vq')
                    quant, _, _ = self.quantize(z)
                    locals()['z_cond'+str(i)] = quant
      # Concatenate along channel dimension
            # z_cond = torch.cat([locals()['z_cond0'],locals()['z_cond1']], dim=1)

            if not self.only_ae: 
                z_cond = locals()['z_cond0']
                original_zcond = locals()['z_cond0']

                # Scale latent representation
                z_cond = z_cond * self.scale

                # Sample noise
                noise = torch.randn_like(locals()['z_cond0'])

                # Perform diffusion sampling
                z_sample = self.sampler(noise, z_cond)

                # Decode sampled latent representation to image space
                y = self.post_quant_conv(z_sample)
            else:
                z_cond = locals()['z_cond0']
                original_zcond = locals()['z_cond0']
                y = self.post_quant_conv(z_cond)
            reconstructions = self.decoder(y)

        return reconstructions,z_cond,original_zcond
    def predict_3d(self, image,cond, patch_size=[112,112,112], overlap=0.5,target=None,mask=None):
        """
        Perform sliding window inference on 3D image
        Args:
            image: 3D numpy array (H,W,D)
            patch_size: size of patches to process
            overlap: overlap ratio between patches
        Returns:
            Reconstructed 3D numpy array
        """
        # Convert to tensor and add batch/channel dimensions
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(self.device)
        cond_tensor = torch.from_numpy(cond).float().unsqueeze(0).to(self.device)
        input_tensor = torch.cat((image_tensor, cond_tensor), dim=1)  # Concatenate along channel dimension
        print('cond_tensor', cond_tensor.shape)

        print('input_tensor', input_tensor.shape)
        with torch.no_grad():
            # Use MONAI's sliding window inference
            reconstruction = sliding_window_inference(
                input_tensor,
                patch_size,
                4,  # batch size
                self.cond_ldm,
                overlap=overlap,
                mode='gaussian',
                sw_device=self.device,
                device=self.device

            )
            recon_ae = sliding_window_inference(image_tensor,patch_size,4,self.ae_model)
        # if target is not None:
        #     # Calculate metrics if target is provided
        #     original_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(self.device)
        #     print('original_tensor', original_tensor.shape,reconstruction.shape)
        #     metrics = self.evaluate(original_tensor, reconstruction)
        #     print(f"SSIM: {metrics[0]}, PSNR: {metrics[1]}")#, f"MI: {metrics[2]}")


        if mask is not None:
            mask_tensor = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            print('mask_tensor',mask_tensor.shape)
        else:
            mask_tensor =None
        if target is not None:
            
            target_tensor = torch.Tensor(target).unsqueeze(0).unsqueeze(0).to(self.device)
            target_tensor = (target_tensor + 1.0) / 2.0
            reconstruction = (reconstruction + 1.0) / 2.0
            metrics = evaluate_metric(target_tensor, reconstruction,mask=mask_tensor)  
            print(f"Image Metrics : SSIM: {metrics[0]}, PSNR: {metrics[1]}")
        
        return reconstruction.cpu().numpy()[0,0],recon_ae.cpu().numpy()[0,0],metrics if target is not None else None






class DM_Inference(VAE_Inference):
    """
    LDM_Inference: Inference class for Latent Diffusion Model (LDM) with VAE encoding and DM-based generation.
    """
    def __init__(self, config,load_fun='state_dict',patch_size=None):
        '''
   
        
        '''
        self.config = config
        self._init_UNet(config)

    
    def _init_UNet(self, config):
        """
        Initialize the UNet model for latent diffusion.
        
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        unet_params = config['model_config']['UNet']
        self.model =  UNet(
            T=config['model_config']['UNet']['T'],
            ch=config['model_config']['UNet']['ch'],
            ch_mult=config['model_config']['UNet']['ch_mult'],
            attn=config['model_config']['UNet']['attn'],
            num_res_blocks=config['model_config']['UNet']['num_res_blocks'],
            dropout=0.1,
            in_ch=config['model_config']['UNet']['in_ch'],
            out_c=config['model_config']['UNet']['out_c']).to(device)
        checkpoint = torch.load(config['model_config']['UNet_checkpoint'], map_location=device)
        self.model.load_state_dict(remove_module_from_state_dict(checkpoint['model']))
        self.model.eval()
        # self.sampler = DDIMSampler(device,n_steps=1000,ddim_step=500,net=self.model,eta=0,infer_T=None,condition_fun='SR3')
 

        self.sampler = SR3DiffusionSampler(self.model, 1e-4, 0.02,T=1000,infer_T=self.config['model_config']['infer_T'])
        self.ddim_model = DDIMSampler(device,n_steps=1000,ddim_step=self.config['model_config']['ddim_steps'],net=self.model,eta=0,infer_T=self.config['model_config']['noiseT'],condition_fun='SR3')
        T = self.config['model_config']['UNet']['T']
        self.ddpm = DDPM(device,T)
 

    def inference(self, cond0):
        """
        Perform inference using the Latent Diffusion Model (LDM) with only the diffusion model.
        
        Args:
            cond0 (torch.Tensor): First conditioning input tensor.
            cond1 (torch.Tensor): Second conditioning input tensor.
        
        Returns:
            torch.Tensor: Generated output tensor.
        """
        self.eval()
        with torch.no_grad():
            # Prepare conditioning tensors
            cond0 = cond0.unsqueeze(0)  # Add batch dimension

            # Encode conditioning inputs
            z_cond0 = self.encoder(cond0)

            # Sample noise
            noise = torch.randn_like(z_cond0)

            # Perform diffusion sampling
            z_sample = self.sampler(noise, z_cond0)

        return z_sample
    def predict_3d(self, T1w,image, patch_size=[96,96,96], overlap=0.5,target=None,mask=None,infer_mode = 'original',prior_noise = 'guassian'):
        """
        Perform sliding window inference on 3D image
        Args:
            image: 3D numpy array (H,W,D)
            patch_size: size of patches to process
            overlap: overlap ratio between patches
        Returns:
            Reconstructed 3D numpy array
        infer_mode: 'original'使用原始采样器，'2-stage'使用两阶段采样器
        prior_noise: 'guassian'使用高斯噪声，'cond'使用cond正向扩散噪声,'img'使用image正向扩散噪声
        """
        # Convert to tensor and add batch/channel dimensions
        cond = T1w
        
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        cond_tensor = torch.from_numpy(cond).float().unsqueeze(0).unsqueeze(0).to(device)
        noise_tensor = torch.randn_like(image_tensor).to(device)
        ##构建正向扩散的先验噪声
        prior_img_noise_tensor = self.ddpm.sample_forward(image_tensor,self.config['model_config']['noiseT'])
        prior_cond_noise_tensor = self.ddpm.sample_forward(cond_tensor,self.config['model_config']['noiseT'])


        if prior_noise == 'guassian':    
            input_tensor = torch.cat((noise_tensor, image_tensor), dim=1)
        elif prior_noise == 'cond':
            input_tensor = torch.cat((prior_cond_noise_tensor, image_tensor), dim=1)
        elif prior_noise == 'img':
            input_tensor = torch.cat((prior_img_noise_tensor, image_tensor), dim=1)
        else:
            raise ValueError("prior_noise must be 'guassian', 'cond', or 'img'")

        with torch.no_grad():
            # Use MONAI's sliding window inference
            if infer_mode == 'original':
                print('original inference')

                reconstruction = sliding_window_inference(
                    input_tensor,
                    patch_size,
                    4,  # batch size
                    self.sampler,
                    overlap=overlap,
                    mode='gaussian',
                    sw_device=self.device,
                    device=self.device

                )
            elif infer_mode == '2-stage':
                pred_mean = 0
                overlap = [0.25]*self.config['model_config']['ddim_steps']
                for i in range(self.config['model_config']['num_ddim']):
                    print('input_tensor',input_tensor.shape)
                    final_asl = sliding_window_inference(input_tensor, (96,96,96), 8, self.ddim_model,overlap=0.25,mode = 'gaussian')
                    pred_mean += final_asl
                coarse_recon = pred_mean / self.config['model_config']['num_ddim']

                input_tensor = torch.cat((coarse_recon,image_tensor),dim=1)
                final_sr = sliding_window_inference(
                    input_tensor,
                    patch_size,
                    4,  # batch size
                    self.sampler,
                    overlap=0.25,
                    mode='gaussian',
                )
                reconstruction = final_sr
                    

                
    


        if mask is not None:
            mask_tensor = torch.Tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
            print('mask_tensor',mask_tensor.shape)
        else:
            mask_tensor =None
        if target is not None:
            target_tensor = torch.Tensor(target).unsqueeze(0).unsqueeze(0).to(device)
            target_tensor = (target_tensor + 1.0) / 2.0
            reconstruction = (reconstruction + 1.0) / 2.0
            metrics = evaluate_metric(target_tensor, reconstruction,mask=mask_tensor)  
            print(f"Image Metrics : SSIM: {metrics[0]}, PSNR: {metrics[1]}")
        
        return reconstruction.cpu().numpy()[0,0], metrics if target is not None else None



class SVR_LDM_Inference(LDM_Inference):
    """
    SVR_LDM_Inference: Inference class for SVR-based Latent Diffusion Model (LDM).
    This class extends LDM_Inference to handle SVR-specific conditioning and inference logic.
    """
    def __init__(self, config, dm_model_dir=None, vae_model_dir=None, mode='vq', load_fun='state_dict', patch_size=None, only_ae=False, cond_fun='concat', device='cuda'):
        super().__init__(config, dm_model_dir, vae_model_dir, mode, load_fun, patch_size, only_ae, device)
        self.double_cond = cond_fun
        self.scale = config['model_config']['z_scale']
        self.config = config

        if self.double_cond in ['gate', 'cross_attention']:
            self.load_fusion_block()

        
        if self.only_ae:
            self.predict_model = AE_UNet_Cond(
                self.encoder, self.decoder, self.quant_conv, self.post_quant_conv,
                self.mode, self.quantize, self.scale, self.double_cond)
        else:
            self.predict_model = AE_Diffusion_Cond(
                self.encoder, self.decoder, self.quant_conv, self.post_quant_conv,
                self.sampler, self.mode, self.quantize , self.scale,
                start_noise='guassian', double_cond=self.double_cond,fusion_block=self.fusion_block)
    
    def load_fusion_block(self):
        if self.double_cond == 'gate':
            self.fusion_block = AdvancedGatedFusionUnit(in_channels=self.config['model_config']['VAE']['z_channels'],
                                                kernel_size=3).to(self.device)
           
        elif self.double_cond == 'cross_attention':
            # self.fusion_block = CrossAttentionFusionBlock(in_channels=self.config['model_config']['VAE']['z_channels'],
            #                                               nuxsm_heads=4,
            #                                               head_dim=32).to(self.device)
            #LinearCrossAttentionFusion
            self.fusion_block = LinearCrossAttentionFusion(in_channels=self.config['model_config']['VAE']['z_channels'],num_heads=4, head_dim=32)



        fusion_block_state = torch.load(self.config['model_config']['UNet_checkpoint'], map_location=self.device)
        ##对fusion_block_state进行处理，当中含有module时去掉
        ##展示fusion block的参数大小
        for key in fusion_block_state['fusion_block']:
            print(f'Key: {key}, Shape: {fusion_block_state["fusion_block"][key].shape}')
        self.fusion_block.load_state_dict(remove_module_from_state_dict(fusion_block_state['fusion_block']))




    def inference(self, cond0, cond1):
        """
        Perform inference using the SVR-based Latent Diffusion Model (LDM).
        
        Args:
            cond0 (torch.Tensor): First conditioning input tensor.
            cond1 (torch.Tensor): Second conditioning input tensor.
        
        Returns:
            torch.Tensor: Generated output tensor.
        """
        self.encoder.eval()
        self.decoder.eval()
        self.quant_conv.eval()
        self.post_quant_conv.eval()
        if self.mode == 'vq' and self.quantize is not None:
            self.quantize.eval()

        with torch.no_grad():
            # Encode conditioning inputs
            z_cond0 = self.quant_conv(self.encoder(cond0))
            z_cond1 = self.quant_conv(self.encoder(cond1))
            if self.mode == 'kl':
                posterior = DiagonalGaussianDistribution(z_cond0)
                z_cond0 = posterior.sample()
                posterior = DiagonalGaussianDistribution(z_cond1)
                z_cond1 = posterior.sample()
            elif self.mode == 'vq' and self.quantize is not None:
                quant0, _, _ = self.quantize(z_cond0)
                quant1, _, _ = self.quantize(z_cond1)
                z_cond0 = quant0
                z_cond1 = quant1

            # Concatenate along channel dimension
            if self.double_cond == 'concat':
                z_cond = torch.cat((z_cond0, z_cond1), dim=1)
            elif self.double_cond == 'add_mean':
                z_cond = (z_cond0 + z_cond1) / 2
            elif self.double_cond == 'add_mean_sqrt':
                z_cond = (z_cond0 + z_cond1) / (2**0.5)
            else:
                raise ValueError("double_cond must be 'concat', 'add_mean', or 'add_mean_sqrt'")

            # Sample noise
            if not self.only_ae:
                noise = torch.randn_like(z_cond0)

                # Perform diffusion sampling
                z_sample = self.sampler(noise, z_cond)
            else:
                z_sample = z_cond

            # Decode sampled latent representation to image space
            y = self.post_quant_conv(z_sample)
            reconstructions = self.decoder(y)

        return reconstructions

    def predict_3d(self, cond0, cond1, patch_size=[112, 112, 112], overlap=0.25, target=None):
        """
        Perform sliding window inference on 3D image using SVR-based LDM.
        
        Args:
            cond0: First conditioning 3D numpy array (H, W, D).
            cond1: Second conditioning 3D numpy array (H, W, D).
            patch_size: Size of patches to process.
            overlap: Overlap ratio between patches.
            target: Optional target 3D numpy array for evaluation.
        
        Returns:
            Reconstructed 3D numpy array.
        """
        # Convert to tensor and add batch/channel dimensions
        print('cond0 shape', cond0.shape, 'cond1 shape', cond1.shape)
        cond0_tensor = torch.from_numpy(cond0).float().unsqueeze(0).unsqueeze(0).to(self.device)
        cond1_tensor = torch.from_numpy(cond1).float().unsqueeze(0).unsqueeze(0).to(self.device)
        input_tensor = torch.cat((cond0_tensor, cond1_tensor), dim=1)  # Concatenate along channel dimension
        print('input_tensor', input_tensor.shape)
        with torch.no_grad():
            # Use MONAI's sliding window inference
            reconstruction = sliding_window_inference(
                input_tensor,
                patch_size,
                1,  # batch size
                self.predict_model,
                overlap=overlap,
                mode='gaussian',
                sw_device=self.device,
                device=self.device
            )

        if target is not None:
            # Calculate metrics if target is provided
            original_tensor = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).to(self.device)
            print('original_tensor', original_tensor.shape, reconstruction.shape)
            metrics = self.evaluate(original_tensor, reconstruction)
            print(f"SSIM: {metrics[0]}, PSNR: {metrics[1]}")  # , f"MI: {metrics[2]}")

        return reconstruction.cpu().numpy()[0, 0]  # Return as numpy array






if __name__ == "__main__":
    config = get_default_config()
    model = DM_Cat_Module(config, vae_type='kl')
    # model.train_loop(total_steps=10000)
    # Test data loader
    train_loader = model.train_loader
    val_loader = model.val_loader

    print("Testing train data loader...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"Image shape: {batch[0].shape}")
        print(f"Condition shape: {batch[1].shape}")
        if i == 2:  # Test first 3 batches
            break

    print("\nTesting validation data loader...")
    for i, batch in enumerate(val_loader):
        print(f"Batch {i+1}:")
        print(f"Image shape: {batch[0].shape}")
        print(f"Condition shape: {batch[1].shape}")
        if i == 2:  # Test first 3 batches
            break
    # Test inference