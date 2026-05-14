import os 
import torch
import h5py
import time
from tqdm import tqdm
import importlib
import  numpy as np
from torch.utils import data
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_one_2D(X,model,slice_channel=7,slide_step =1,aug=False):
    '''
    用于将3D图像转换为2D堆栈图像，从而使用2D模型进行预测
    '''
    # print('inference_2D slide_step',slide_step)
    model=model.eval()
    ##test_aug

    with torch.no_grad():
        x0=X[np.newaxis,:,:,:]
        x0=torch.tensor(x0).type(torch.FloatTensor).to(device)
        

        slice_len =X.shape[0]
        ##使得可以整除：
        infer_num = (slice_len - slice_channel)//slide_step + 1##即便可以整除也加一 213
        infer_len =slide_step*infer_num +slice_channel
        img = torch.zeros([1,infer_len,x0.shape[-2],x0.shape[-1]]).to(device)
        img[0,:slice_len,:,:]=x0
        target =torch.zeros([1,infer_len,x0.shape[-2],x0.shape[-1]]).to(device) ##[1,646,256,256]
        pro_mask =torch.zeros([1,infer_len,x0.shape[-2],x0.shape[-1]]).to(device)  ##pro_mask用于计算重叠部分

        

        for  i in range(infer_num +1):

            tmp = img[:,i*slide_step:i*slide_step+slice_channel,:,:].to(device) 
            target[:,i*slide_step:i*slide_step+slice_channel,:,:] += model(tmp)
            pro_mask[:,i*slide_step:i*slide_step+slice_channel,:,:] += torch.ones([1,slice_channel,x0.shape[-2],x0.shape[-1]]).to(device)

        target_final = target/pro_mask

        result =np.squeeze(target_final.cpu().numpy())
        result =result[:slice_len]
    return result