# calculate inception score in numpy
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import exp
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import torch
from torchvision.models import inception_v3
# from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import cv2
from torchvision.transforms import ToTensor
import torch.nn as nn
import sys
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision import transforms
# calculate the inception score for p(y|x)
def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = mean(sum_kl_d)
    # undo the logs
    is_score = exp(avg_kl_d)
    return is_score

# conditional probabilities for low quality images
# p_yx = asarray([[0.33, 0.33, 0.33], [0.33, 0.33, 0.33], [0.33, 0.33, 0.33]])
# score = calculate_inception_score(p_yx)
# print(score)
# def inception_score(images, batch_size=32, splits=10):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = inception_v3(pretrained=True, transform_input=False).to(device)
#     model.eval()

#     scores = []
#     num_batches = len(images) // batch_size

#     with torch.no_grad():
#         for i in range(num_batches):
#             batch = torch.stack([img.to(device) for img in images[i*batch_size:(i+1)*batch_size]])
#             print('batch',batch.shape)
#             preds = nn.Softmax(dim=1)(model(batch))
#             print('pred',preds.shape)
#             p_yx = preds.log()
#             p_y = preds.mean(dim=0).log()
#             print('p_yx',p_yx.shape,p_y.shape)
#             kl_divergence = torch.sum(p_yx * (p_yx - p_y), dim=1).mean()

#             scores.append(torch.exp(kl_divergence))
#     print('scores',scores)
#     scores = torch.stack(scores)
#     mean_score = scores.mean()
#     std_score = scores.std()
#     return mean_score.item(), std_score.item()





from tool.image_process import save_nifit
def inception_score(images,batch_size=50, resize=False, splits=10):
    eps=1E-16
    # Set up dtype
    device = torch.device("cuda:0")  # you can change the index of cuda
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    
    def get_pred(x):
        with torch.no_grad():
            if resize:
                x = up(x)
            x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
    

    N = len(images) 
    num_batches = len(images) // batch_size

    preds = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
                 'Setting batch size to data size'))
    for i in range(num_batches):
        batch = torch.stack([img.to(device) for img in images[i*batch_size:(i+1)*batch_size]])
        preds[i*batch_size :(i+1)*batch_size] = get_pred(batch)
        

    # assert batch_size > 0
    # assert N > batch_size

    # Now compute the mean KL Divergence
    # print('Computing KL Divergence')
    # print('preds',preds.shape)
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            # print('py',py)
            # print('entropy',entropy(pyx, py))

            # kl_d = pyx * (log(pyx + eps) - log(py + eps))
            # avg_kl_d = mean(kl_d)
            # print('avg_kl_d',avg_kl_d)
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(scores))

    return np.max(split_scores), np.mean(split_scores),np.std(split_scores)
    # return np.mean(split_scores)



def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)*2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)
    return fid



def fid_score_test(images_t,images_g,channel3=True):
    '''
    参考https://blog.csdn.net/qq_42443342/article/details/132586590
    channel3输入是否是按照inception_v3的三通道设置
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False)
    
    # 将通道数改为1
    if channel3:
        model = model
    else:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # 删除最后一层全连接层
    model.fc = nn.Sequential()
    model.to(device)
    model=nn.DataParallel(model,device_ids=[0,1])
    model.eval()
 
 
    with torch.no_grad():
        act1 = model(images_t).detach().cpu()
        act2 = model(images_g).detach().cpu()
    act_values2 = act2.numpy()
    act_values1 = act1.numpy()
    mu1, sigma1 = act_values1.mean(axis=0), np.cov(act_values1, rowvar=False)
    mu2, sigma2 = act_values2.mean(axis=0), np.cov(act_values2, rowvar=False)
    #
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
 
    # calculate sqrt of product between cov, sqrtm( ) 对矩阵整体开平方
    # print('sigma1.dot(sigma2)',sigma1.dot(sigma2))
    covmean = sqrtm(sigma1.dot(sigma2))
 
    # check and correct imaginary numbers from sqrt,如果covmean中有复数，则返回true
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    del model
    return fid




def fid_score_test2(images_t,images_g,channel3=True):
    '''
    参考https://blog.csdn.net/qq_42443342/article/details/132586590
    channel3输入是否是按照inception_v3的三通道设置
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False)
    batch_size = 100
    # 将通道数改为1
    if channel3:
        model = model
    else:
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    # 删除最后一层全连接层
    model.fc = nn.Sequential()
    model.to(device)
    # model=nn.DataParallel(model,device_ids=[0,1])
    model.eval()

    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])  
 
    with torch.no_grad():
        images_t = transform(images_t).to(device)
        images_g = transform(images_g).to(device)
        images_t = images_t.to(device)
        images_g = images_g.to(device)

        act1 = torch.zeros([1,2048])
        act2 = torch.zeros([1,2048])
        for n in range(images_t.shape[0]//batch_size +1):
            images_tb = images_t[n*batch_size:(n+1)*batch_size]
            images_gb = images_g[n*batch_size:(n+1)*batch_size]
            # print('images_t',images_t.shape)
            act1 = torch.cat([act1,model(images_tb).detach().cpu()],dim=0)
            act2 = torch.cat([act2,model(images_gb).detach().cpu()],dim=0)
        act1 = act1[1:]
        act2 = act2[1:]


    act_values2 = act2.numpy()
    act_values1 = act1.numpy()
    mu1, sigma1 = act_values1.mean(axis=0), np.cov(act_values1, rowvar=False)
    mu2, sigma2 = act_values2.mean(axis=0), np.cov(act_values2, rowvar=False)

    #
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean =sqrtm(sigma1.dot(sigma2))

 
    # check and correct imaginary numbers from sqrt,如果covmean中有复数，则返回true
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    del model
    return fid


# Load the CLIP model
# model_ID = "clip-vit-base-patch16"
# model = CLIPModel.from_pretrained(model_ID)

# preprocess = CLIPImageProcessor.from_pretrained(model_ID)


# Define a function to load an image and preprocess it for CLIP
def load_and_preprocess_image(image_path):
    model_ID = "clip-vit-base-patch16"
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)
    # Load the image from the specified path
    image = Image.open(image_path)
    # Apply the CLIP preprocessing to the image
    image = preprocess(image, return_tensors="pt")
    # Return the preprocessed image
    return image


def clip_img_score(img_a,img_b):

    # Load the CLIP model
    model_ID = "clip-vit-base-patch16"
    preprocess = CLIPImageProcessor.from_pretrained(model_ID)
    # Load the image from the specified path
    
    model = CLIPModel.from_pretrained(model_ID)




    preprocess = CLIPImageProcessor.from_pretrained(model_ID)
    # Load the two images and preprocess them for CLIP


    img_a = Image.fromarray(np.uint8(img_a*255))
    img_b = Image.fromarray(np.uint8(img_b*255))

    image_a = preprocess(img_a, return_tensors="pt")
    image_b = preprocess(img_b, return_tensors="pt")
    image_a = image_a["pixel_values"]
    image_b = image_b["pixel_values"]




    # image_a = load_and_preprocess_image(img1_path)["pixel_values"]
    # image_b = load_and_preprocess_image(img2_path)["pixel_values"]

    # Calculate the embeddings for the images using the CLIP model
    with torch.no_grad():
        embedding_a = model.get_image_features(image_a)
        embedding_b = model.get_image_features(image_b)

    # Calculate the cosine similarity between the embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding_a, embedding_b)
    return similarity_score.item()

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def Caculate_IS_FID(data_x,data_y,data_mask):
        x = torch.Tensor(data_x[np.newaxis,np.newaxis]).cuda()#.to(device)
        y = torch.Tensor(data_y[np.newaxis,np.newaxis]).cuda()
        input_x = []
        input_y = []
        for i in range(data_x.shape[2]-3):
            # print(np.sum(data_mask[:,:,i]))
            if np.sum(data_mask[:,:,i])>=0:    
                input_x.append(x[0,0,:,:,i:3+i].permute(2,0,1))
                input_y.append(y[0,0,:,:,i:3+i].permute(2,0,1))
        with HiddenPrints():
                is_score = inception_score(torch.stack(input_y),batch_size=10,splits=20)
                fid_score = fid_score_test(torch.stack(input_x),torch.stack(input_y),channel3=True)
        return is_score,fid_score