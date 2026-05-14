import os
import sys
from tkinter import image_names
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim

def add_noise(input_patch, SNR_noise=10,SNR_lb=5., SNR_ub=20.):##pro设置概率
    # max_lr = np.max(input_patch)
    # input_patch[input_patch<0] = 0
 
    # input_patch = input_patch/max_lr


    # time0=time.time()
    # if np.random.uniform() < pro:
        # https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    if SNR_noise == None:
        Power_ratio = np.random.uniform(10. ** (SNR_lb / 10.), 10. ** (SNR_ub / 10.)) # SNR: 5dB ~ 20dB
    else:
        Power_ratio = 10. ** (SNR_noise / 10.)
    print(Power_ratio,10*np.log10(Power_ratio)) ##取固定噪声
    sigma = np.linalg.norm(input_patch) / np.sqrt(input_patch.size * Power_ratio) ##np.linalg.norm默认2范数，即根号下x平方和
    print('sigma',sigma)
    noise = np.random.normal(size=input_patch.shape)
    print(np.mean(noise),np.std(noise))
    input_noise = input_patch + sigma * noise


    SNR=10*np.log10(np.linalg.norm(input_patch)**2/np.linalg.norm(sigma * noise)**2)
    print('SNR',SNR)
    # print('add noise time',time.time()-time0,'s')
    # input_patch = input_patch*max_lr


    return input_noise,sigma * noise


def add_noise_tensor(input_patch, SNR_noise=10):
    Power_ratio = 10. ** (SNR_noise / 10.)
    input_numpy =input_patch.cpu().numpy()
    sigma = np.linalg.norm(input_numpy) / np.sqrt(input_numpy.size * Power_ratio) 
    noise = torch.randn_like(input_patch)

    input_noise = input_patch + sigma * noise
    return input_noise
