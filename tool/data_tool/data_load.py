import numpy as np
import scipy.ndimage as ndimage
import time
import torch
import torch.nn as nn
from torch.utils import data
import h5py
import os
import nibabel as nib
from scipy.ndimage import zoom
import copy
from tool.image_process import crop_pad3D
from matplotlib import pyplot as plt
def normalize_mri(image):
    # Z-score 标准化
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = div0((image - mean), std)

    # 线性映射到 [-1, 1]
    normalized_image = 2 * (normalized_image - np.min(normalized_image)) / np.ptp(normalized_image) - 1

    return normalized_image

def load_nifit(data_path):
    img = nib.load(data_path)
    tmp = np.squeeze(img.get_fdata()).astype(np.float32)
    return tmp

def save_nifit(data, filename):
    # print(data.dtype)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)

def idsplit(n_subject, ratio, shuffle=True):
    '''
    split n_subject data,
    if ratio<1, then train_id=n*ratio,
    if ratio>1, then cross-validaiton
    '''
    id_list = np.arange(n_subject)
    if shuffle:
        np.random.shuffle(id_list)
    if ratio <= 1:
        n = int(np.round(n_subject*ratio))
        train_id = id_list[:n]
        test_id = id_list[n:]
    else:
        train_id = [None]*ratio
        test_id = [None]*ratio
        add_one = n_subject % ratio
        tmp = id_list[add_one:].reshape((ratio, -1))
        c = 0
        # print(id_list[:add_one])
        for n in range(ratio):
            list_one = id_list[:add_one].tolist()
            test_id[n] = tmp[n].tolist()
            if c < add_one:
                test_id[n].append(list_one.pop(c))
                c += 1
            train_id[n] = np.delete(tmp, n, 0).flatten().tolist()
            train_id[n] += list_one

    return train_id, test_id


def normlize_data(tmp):#用于归一化到-1,1
    a_min = np.min(tmp)
    a_max = np.max(tmp)
    tmp = 2*(tmp -a_min)/(a_max- a_min)-1
    return tmp

def normlize_mean_std(tmp):
    tmp_std = np.std(tmp)
    tmp_mean = np.mean(tmp)
    # tmp = (tmp - tmp_mean) / tmp_std
    tmp = div0(tmp - tmp_mean, tmp_std)
    return tmp


def div_mean(tmp,num_dim=True):
    'num_dim:输入的是单个数据还是个别数据:True:是一组数据'
    if num_dim:
        data =[]
        for i in range(tmp.shape[0]):
            mean_value = np.mean(tmp[i])
            data_tmp = tmp[i]/mean_value
            data.append(data_tmp)
        data =np.array(data)
    else:
        data=tmp/np.mean(tmp)

    return data


def div0(a, b):
    if b == 0:
        c = np.zeros_like(a)
    else:
        c = a / b
    return c



class Patch_data_multi(data.IterableDataset):##
    '这个函数使用路径来生成数据,而不是矩阵'  
    
    def __init__(self, data_path,
                        data_name, 
                        all_data = False,
                        patch_size=[96,96,96], 
                        stride=[1,1,1],
                        trainable =True,
                        data_prenorm ='div_mean',
                        augmentation=False,
                        patch_training=True,
                        is_hot_label=False,
                        data_norm=False,
                        channel_first=False,
                        n_labels=3,
                        label_id = False,
                        mask_name = 'mask',
                        output_fname= False,
                        patch_pro = 0.7,
                        small_target_labels = None,
                        ct_hu=None):##patch_training=False用于val时取完整图像
        'data_norm: 一般用于diffusion model的正则化:[-1,1]'
        'all_data:是否使用所有数据，如果是False,则使用交叉验证'
        'channel_first:人为设置通道在最后还是在前面,默认False:在最后面,这样方便使用一些代码时灵活测试'
        'data_name:存储每个case的hdf5时往往需要多存储许多'
        'trainable选择是输出训练集还是验证集,True训练，False验证'
        """
        data_prenrom:一般要对图像进行标准化：选择：'div_mean':除以平均值。使得输入的值范围不至于过大
                                                'z_score':进行Z-socre标准化   
                                                'z_score_norm': 进行Z-socre标准化后归一到[-1,1],常用于MRI 比如函数normalize_mri 
                                                'CT_hu':针对CT的hu值进行归一化到[-1,1]
        ct_hu:针对CT的hu值进行归一化到[-1,1],需要设置CT的hu值范围，比如ct_hu=[-200, 400]
        label_id:对于复杂的任务，有可能需要分类的标签。默认为Flase,注意默认需要把分类Id放在最后面 
        mask_name:默认为mask,如果有mask的话，可以选择mask的名字,可以设置其他名字用于分割
        output_fname:输入图像的fname,以便于对训练数据进行查看，或者直接根据文件名设置分类
        patch_pro:patch内包含非0区域的比例，可条件，默认0.7
        target_label:  list. 指定只读取某种特定标签的数据。例如 target_label=[1] 或 target_label=[0, 2]
        """
        self.data_prenorm = data_prenorm
        self.ct_hu = ct_hu
        fold=4
        self.trainable = trainable
        self.data_path = data_path
        self.data_name = data_name
        self.patch_pro = patch_pro


        ## NEW: 保存目标标签
        self.small_target_labels = small_target_labels
        ##判断data_path是否为list
        if isinstance(self.data_path, list):
            self.file_list = []

            ##求子文件最多那个路径的子文件数目：
            max_num = 0
            for tmp_path in self.data_path:
                print('tmp_path',tmp_path)
                file_list = sorted(os.listdir(tmp_path))
                max_num = max(max_num, len(file_list))
            print('max_num',max_num)
    
            for tmp_path in data_path:
                file_list = sorted(os.listdir(tmp_path))
                file_list = [os.path.join(tmp_path, file) for file in file_list]
                ##当前路径的文件数目小于最大路径的文件数目时，进行补齐
                if len(file_list) < max_num:
                    for i in range(max_num - len(file_list)):
                        file_list.append(file_list[i % len(file_list)])
                self.file_list += file_list
            print('len',len(self.file_list))
        else:
            self.file_list = sorted(os.listdir(self.data_path))
            self.file_list = [os.path.join(self.data_path, file) for file in self.file_list]

        
            

        # print('selffile_list',self.file_list)
        
        num_file =  len(self.file_list)
        train_id,val_id  = idsplit(num_file,5,shuffle=False)
        if all_data:
            train_id = np.arange(num_file)
            val_id = np.arange(num_file)
            self.train_file_list = self.file_list
            self.val_file_list = self.file_list
        else:
            self.train_file_list = []
            self.val_file_list = []
            for id in train_id[fold]:
                self.train_file_list.append(self.file_list[id])
            for id in val_id[fold]:
                self.val_file_list.append(self.file_list[id])
        self.patch_size = patch_size
        self.label_id = label_id

        ##输入train_file_list和val_file_list
        # print('train_file_list',self.train_file_list)


        # print('val_file_list',self.val_file_list)
        


        

        # print('train_file_list',self.train_file_list)
        # print('val_file_list',self.val_file_list)
        self.channel_first =channel_first
        
        
 

        ##Data_Y应该是一个list ,便于使用不同的数量的数据，Data_Y=[Y0] or  
        
       
        self.is_hot_label=is_hot_label # int(np.max(Y[0])+1)###注意加int
        # self.patch_edge = self.__calc_edge__()
        self.aug=augmentation
        self.patch=patch_training
        self.data_norm =data_norm
        self.n_labels =n_labels
        self.mask_name = mask_name
        self.output_fname = output_fname


    def get_file_list(self):
        """
        返回训练和验证文件列表
        """
        return self.train_file_list, self.val_file_list
    def __get_patch(self, X, start_pos, patch_size):
        Y = X[start_pos[0]:start_pos[0] + patch_size[0],
            start_pos[1]:start_pos[1] + patch_size[1],
            start_pos[2]:start_pos[2] + patch_size[2]]
        return Y 
    
    def __iter__(self):
        # np.random.shuffle(self.patch_index)

        
        while True:
            if self.trainable:
                n_subject=len(self.train_file_list) 
                selected_subject = np.random.randint(n_subject)
                fname= self.train_file_list[selected_subject]
            else:
                n_subject=len(self.val_file_list) 
                selected_subject = np.random.randint(n_subject)
                fname= self.val_file_list[selected_subject]

            hdf5_path = fname #self.data_path+'/'+str(fname)

            self.data_list = []
            time0 =time.time()

            with h5py.File(hdf5_path, "r") as hf:
                key_list = list(hf.keys())
                ##如果有任何一个data_name不在key_list中，则跳过
                data_name_exist = True
                for data_name in self.data_name:
                    if data_name not in key_list:
                        data_name_exist = False
                        break
                if not data_name_exist:
                    print('跳过文件，因为data_name不完整:',hdf5_path)
                    continue
                self.X = hf[str(self.data_name[0])][()] ##主存储数据，用于提取patch

                ##用于生成mask
                x_mask =  self.X.copy()
                eps = 1e-2
                x_mask[x_mask>eps] = 1
                x_mask[x_mask<=eps] = 0

                self.X_mask = np.array(x_mask,'int32')
                ##如果是有target_label的话，则需要进行筛选，生成新的mask,只保留target_label的部分,从而解决数据不平衡的问题,一般用于分类问题
                if self.small_target_labels is not None:
                    self.gt = hf[str(self.mask_name[0])][()] if self.mask_name[0] in hf else hf[str(self.data_name[-1])][()]
                    x_mask = np.zeros_like(self.gt)
                    for label in self.small_target_labels:
                        x_mask[self.gt == label] = 1
                    self.X_mask = np.array(x_mask,'int32')
                ####



                for i in range(len(self.data_name)):
                    tmp = hf[str(self.data_name[i])][()]
                    


                    if self.data_name[i] in self.mask_name:
                        tmp = tmp 
                    else:
                        if self.data_prenorm == 'div_mean':
                            #tmp = div_mean(tmp)##不能用这个，这个是针对数据组的，第一个维度是case
                            tmp = tmp/np.mean(tmp)
                        elif self.data_prenorm == 'z_score':
                            tmp = normlize_mean_std(tmp)
                        elif self.data_prenorm == 'z_score_norm':
           
                            tmp = normalize_mri(tmp)
                        elif self.data_prenorm == 'CT_hu':
                            # print('CT_hu归一化处理')
                            if self.ct_hu is not None:
                                hu_min = self.ct_hu[0]
                                hu_max = self.ct_hu[1]
                                tmp = np.clip(tmp, hu_min, hu_max)
                                tmp = 2 * (tmp - hu_min) / (hu_max - hu_min) - 1
                        else:
                            tmp = tmp
                    self.data_list.append(tmp)
            

            ##如果有label_id,则把label_id取出来
            if self.label_id:
                self.data_list = self.data_list[0:-1]
                label_name = self.data_list[-1]
                ##label_name的处理，之后再说
        
            
            # print('read hdf5',time.time()-time0,'s')

            train_data=[]
            if self.patch_size[0]>self.X.shape[0] :
                for i in range(len(self.data_list)):
                    self.data_list[i] = np.pad(self.data_list[i],((0,self.patch_size[0]-self.X.shape[0]),(0,0),(0,0)),'constant')
           
            if self.patch_size[1]>self.X.shape[1] :
                for i in range(len(self.data_list)):
                    self.data_list[i] = np.pad(self.data_list[i],((0,0),(0,self.patch_size[1]-self.X.shape[1]),(0,0)),'constant')
                   
            if self.patch_size[2]>self.X.shape[2] :
                for i in range(len(self.data_list)):
                    self.data_list[i] = np.pad(self.data_list[i],((0,0),(0,0),(0,self.patch_size[2]-self.X.shape[2])),'constant')
                 
      

            image_size=np.array(self.data_list[i]).shape
            # self.n_pixels  = image_size[0] * image_size[1] *image_size[2]
            self.n_pixels  = np.prod(self.patch_size)
            n0 = image_size[0] - self.patch_size[0] + 1
            n1 = image_size[1] - self.patch_size[1] + 1
            n2 = image_size[2] - self.patch_size[2] + 1
            index = np.random.randint((n0,n1,n2))
            ##如果有small_target_labels的话，则需要根据mask为中心来选取patch,确定mask_n0_start, mask_n1_start, mask_n2_start,mask_n0_end,mask_n1_end,mask_n2_end
            if self.small_target_labels is not None:
                mask_n0,mask_n1,mask_n2 = np.where(self.X_mask>0)
                mask_n0_start = np.min(mask_n0)
                mask_n0_end = np.max(mask_n0)
                mask_n1_start = np.min(mask_n1)
                mask_n1_end = np.max(mask_n1)
                mask_n2_start = np.min(mask_n2)
                mask_n2_end = np.max(mask_n2)
                index = np.random.randint((mask_n0_start,mask_n1_start,mask_n2_start),
                                        (max(mask_n0_end - self.patch_size[0] +1, mask_n0_start+1),
                                        max(mask_n1_end - self.patch_size[1] +1, mask_n1_start+1),
                                        max(mask_n2_end - self.patch_size[2] +1, mask_n2_start+1)
                                        ))
                # print('index',index)




            patch_probability = 0
            if self.patch:
                extra_patch_index = 0
                while patch_probability<=self.patch_pro:
                    X_patch = self.__get_patch(np.array(self.X), index, self.patch_size)
                    X_mask_patch = self.__get_patch(np.array(self.X_mask), index, self.patch_size)
                    # patch_probability = np.sum((X_mask_patch>0).astype(np.float))/(self.n_pixels)
                    # print('n_pixels',self.n_pixels)
                    patch_probability = np.sum(X_mask_patch.astype(np.float32)) /(self.n_pixels) #/(self.n_pixels).astype(np.float)/0.1
                    # print('patch_probability',patch_probability)
                    # img = X_patch[int(self.patch_size[0]/2),:,:]
                    ##save png:
                    # plt.imsave('/data/yunzhixu/medical_fusion/hydro_source/tmp_data/'+str(patch_probability)+'.png', img)
                    extra_patch_index += 1
                    ##判断取次数，过多则直接跳过
                    if extra_patch_index>10:
                        break
                # print('get')
                ##
                # print('final patch_probability',patch_probability,self.patch_pro)
                if patch_probability>self.patch_pro : #np.random.uniform():#np.random.uniform() < patch_probability:##起码patch_probability必须大于0或者说不能太小，才能使用patch
                    
                    if self.data_norm:

                        for i in range(len(self.data_list)):
                            tmp =  normlize_data(self.data_list[i])
                        
                            
                            tmp_patch = self.__get_patch(np.array(tmp), index, self.patch_size)
                            #判断tm_path的大小和patch_size的大小对得上吗
                            if tmp_patch.shape != self.patch_size:
                                tmp_patch = crop_pad3D(tmp_patch, self.patch_size)
                            train_data.append(tmp_patch)
                    else:     

                        for i in range(len(self.data_list)):
                            tmp = self.data_list[i]
                      
                            tmp_patch = self.__get_patch(np.array(tmp), index, self.patch_size)
                            if tmp_patch.shape != self.patch_size:
                                tmp_patch = crop_pad3D(tmp_patch, self.patch_size)
                            train_data.append(tmp_patch)

                    
                    '于加入分割任务的标签比较麻烦,直接默认为Y是data_list[1]'
                    new_data = []
                    for i in range(len(train_data)):
                        tmp = train_data[i]
                        tmp = tmp[:,:,:,np.newaxis]
                        new_data.append(tmp)
                    train_data =new_data
                   
                    if self.is_hot_label:##判断是否需要one_hot的分割任务
                        Y_patch = np.squeeze(train_data[1],axis=-1)
                        Y_patch=torch.tensor(Y_patch).type(torch.int64)
                        Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels).numpy()
                        train_data[1] = Y_patch

                    a=time.time()
                   
                    if self.aug:
                        train_data = flip_lr_list(train_data)
                        new_data = []
                        for n in range(len(train_data)):
                            tmp = torch.tensor(train_data[n])
                            new_data.append(tmp)
                        
                        train_data = rot_3d_torch_list(new_data)
                        train_data = shift_3d_int_list(train_data)
                    final_data = []
                    for n in range(len(train_data)):
                        tmp = torch.tensor(train_data[n]).type(torch.FloatTensor)
                        if self.channel_first:
                            tmp=tmp.permute(3,0,1,2)
                        
                        final_data.append(tmp)
                    if self.output_fname: ##判断是否需要输出文件路径名
                        yield final_data,fname
                    else:
                        yield final_data
            else:

                train_data =[]
                for i in range(len(self.data_list)):
                    tmp = self.data_list[i]
                    tmp = tmp[:,:,:,np.newaxis]
                    train_data.append(tmp)

      
                # if self.is_hot_label:
                #     Y_patch=torch.tensor(Y_patch).type(torch.int64)
                #     Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels)
                # else:
                #     Y_patch = Y_patch[:,:,:,np.newaxis]


                if self.aug:

                    train_data = flip_lr_list(train_data)
                    new_data = []
                    for n in range(len(train_data)):
                        tmp = torch.tensor(train_data[n])
                        new_data.append(tmp)

                    train_data = rot_3d_torch_list(new_data)
                    train_data = shift_3d_int_list(train_data)
                final_data = []
                for n in range(len(train_data)):
                    tmp = torch.tensor(train_data[n]).type(torch.FloatTensor)
                    if self.channel_first:
                        tmp=tmp.permute(3,0,1,2)
                    final_data.append(tmp)

                if self.output_fname: ##判断是否需要输出文件路径名
                    yield final_data,fname
                else:
                    yield final_data














    
###用于aug3D的函数：

def rot_3d_torch_list(data_list):##一定要写if ，不然np跳过后返回的是一个空列表
    data = []
    if np.random.uniform() > 0.5:
        axis = ((0,1),(0,2),(1,2))
        #t=time.time()
        for i in range(len(data_list)):
            tmp = data_list[i]
            for n in range(len(axis)):
                tmp=torch.rot90(tmp,1,axis[n])
            data.append(tmp)
        
    return data_list

def shift_3d_int_list(data_list, max_shift=10):
    data = []
    if np.random.uniform() > 0.5:
        
        random_shift = np.random.randint(-max_shift, max_shift, size=3)
        for i in range(len(data_list)):
            tmp = data_list[i]
            tmp = torch.roll(tmp, list(random_shift), dims=(0,1,2))
            data.append(tmp)
        data_list =data
    return data_list


def rot_3d_torch(X, Y):
    if np.random.uniform() > 0.5:
        axis = ((0,1),(0,2),(1,2))
        #t=time.time()
        for n in range(len(axis)):
            X=torch.rot90(X,1,axis[n])
            Y=torch.rot90(Y,1,axis[n])
    return X,Y
    
def flip_lr_list(data_list):##用于实现对有多个数据的list
    data=[]
    if np.random.uniform() > 0.5:
        for i in range(len(data_list)):
            tmp = data_list[i][::-1,:,:,:].copy()
            data.append(tmp)
        data_list =data
    return data_list
    

def flip_lr(X, Y):
    if np.random.uniform() > 0.5:
        X = X[::-1,:,:,:].copy()
        Y = Y[::-1,:,:,:].copy()####加copy()防止报错
    return X, Y

def rot_3d(X, Y, max_angle=15):
    axis = ((0,1),(0,2),(1,2))
    #t=time.time()
    for n in range(len(axis)):
        theta = np.random.uniform(-max_angle, max_angle)
        X = ndimage.rotate(np.squeeze(X), theta, axes=axis[n], reshape=False, mode='reflect')
        Y = ndimage.rotate(np.squeeze(Y), theta, axes=axis[n], reshape=False, mode='reflect', order=1)
    #return X[:,:,:,np.newaxis], (Y>0.5).astype(int)[:,:,:,np.newaxis]####对于多标签的情况Y>0.5肯定是有大问题的
    ##报错出在这，因为已经把onehot改在了前面，那么这里就是多了维度的不用np.newaxis
    #print(time.time()-t,'s')
    return X[:,:,:,np.newaxis], (Y>0.5).astype(int)##X[:,:,:,np.newaxis]还是要加上，因为有一个np.squeeze(X)，会把维度1给变没


def shift_3d(X ,Y, max_shift=10):
    t=time.time()
    #print("shift_3d",X.shape,Y.shape)
    random_shift = np.random.uniform(-max_shift,max_shift,4)
    random_shift[3]=0 # channel demension 
    print(time.time()-t,'s')
    X = ndimage.shift(X, shift=random_shift, mode='reflect')
    print(time.time()-t,'s')
    Y = ndimage.shift(Y, shift=random_shift, mode='reflect', order=1)
    print(time.time()-t,'s')
    #print("shift_3d输出",X.shape,Y.shape)
    return X, (Y>0.5).astype(int)


def shift_3d_int(X ,Y, max_shift=10):
    if np.random.uniform() > 0.5:
        random_shift = np.random.randint(-max_shift, max_shift, size=3)
        X = torch.roll(X, list(random_shift), dims=(0,1,2))
        Y = torch.roll(Y, list(random_shift), dims=(0,1,2))
    return X, Y





