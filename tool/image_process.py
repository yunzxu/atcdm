import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom
import h5py
import SimpleITK as sitk

from matplotlib import pyplot as plt

def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled

def resampleVolume(outspacing, vol,resamplemethod=sitk.sitkLinear):
    """
    将体数据重采样的指定的spacing大小\n
    paras：
    outpacing：指定的spacing，例如[1,1,1]
    vol：sitk读取的image信息，这里是体数据\n
    return：重采样后的数据
    """
    outsize = [0, 0, 0]
    # 读取文件的size和spacing信息
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
 
    transform = sitk.Transform()
    transform.SetIdentity()
    # 计算改变spacing后的size，用物理尺寸/体素的大小
    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
 
    # 设定重采样的一些参数
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(resamplemethod)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol


def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()      # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()   # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()      # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)		# 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)   # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))    
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled


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


def load_nifit(data_path):
    img = nib.load(data_path)
    tmp = np.squeeze(img.get_fdata()).astype(np.float32)
    return tmp


def save_nifit(data, filename):
    # print(data.dtype)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)


def normlize_mean_std(tmp):
    tmp_std = np.std(tmp)
    tmp_mean = np.mean(tmp)
    # tmp = (tmp - tmp_mean) / tmp_std
    tmp = div0(tmp - tmp_mean, tmp_std)
    return tmp


def normlize_min_max(tmp):
    tmp_max = np.amax(tmp)
    tmp_min = np.amin(tmp)
    tmp = (tmp - tmp_min) / (tmp_max - tmp_min)
    return tmp


def div0(a, b):
    if b == 0:
        c = np.zeros_like(a)
    else:
        c = a / b
    return c


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


def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))#np.ceil向上取整
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)##当目标更大时，填充情况因为向上取值X1有可能会比目标大小大1(0.5向丄取整)，就向前填补坐标
    start_pos = start_pos.astype(int)##若为裁剪，相差基数，向上取整，相当于前面多裁剪1 如17/2=9  9:xx 0到8被裁剪，即前面裁剪9
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    
    return y ##这里不返回所有，建议新写一个函数返回，以保证之前的代码不报错


def crop_pad3D_index(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))#np.ceil向上取整
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)##当目标更大时，填充情况因为向上取值X1有可能会比目标大小大1(0.5向丄取整)，就向前填补坐标
    start_pos = start_pos.astype(int)##若为裁剪，相差基数，向上取整，相当于前面多裁剪1 如17/2=9  9:xx 0到8被裁剪，即前面裁剪9
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    #返回一个坐标，对于裁剪后填充回去比较方便 分类讨论：是裁剪处理时，start_pos可以直接用于填充回原始图像
    #当是填充处理时，start_pos要么0要么1，需要pad_size 即原始图像前面填充pad_size-start_pos
    return y,start_pos,pad_size





def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))#np.ceil向上取整
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)##当目标更大时，填充情况因为向上取值X1有可能会比目标大小大1(0.5向丄取整)，就向前填补坐标
    start_pos = start_pos.astype(int)##若为裁剪，相差基数，向上取整，相当于前面多裁剪1 如17/2=9  9:xx 0到8被裁剪，即前面裁剪9
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    
    return y ##这里不返回所有，建议新写一个函数返回，以保证之前的代码不报错


def crop_pad3D_index(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))#np.ceil向上取整
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)##当目标更大时，填充情况因为向上取值X1有可能会比目标大小大1(0.5向丄取整)，就向前填补坐标
    start_pos = start_pos.astype(int)##若为裁剪，相差基数，向上取整，相当于前面多裁剪1 如17/2=9  9:xx 0到8被裁剪，即前面裁剪9
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    #返回一个坐标，对于裁剪后填充回去比较方便 分类讨论：是裁剪处理时，start_pos可以直接用于填充回原始图像
    #当是填充处理时，start_pos要么0要么1，需要pad_size 即原始图像前面填充pad_size-start_pos
    return y,start_pos,pad_size


def onehot(Y,n_label):
    Y=torch.tensor(Y).type(torch.int64)
    Y=torch.nn.functional.one_hot(Y, num_classes=n_label)###因为要做采样建议onehot分开使用
    Y=np.array(Y)
    return Y





#X0=np.array(X,dtype=np.float32)
# X_final=np.zeros((80,128,128,128))#固定尺寸
# Y_final=np.zeros((80,128,128,128))

def crop_edge_sample(X,Y,resample=[144,144,128],n_label=10):###取全脑区域并完成采样，L保存每个图像（对应编号）的全脑大小
    X0=np.array(X)
    X0=(X0>0)
    shape0=[]
    X_final=[]
    Y_final=[]
    L=[]
    shapex,shapey,shapez=X[0].shape
    index=[] ##上下左右前后6个方位的裁剪值，方便还原
    for i in range(X.shape[0]):
        X1=np.sum(X0[i],axis=(1,2))#x轴
        X2=np.sum(X0[i],axis=(0,2))#y轴
        X3=np.sum(X0[i],axis=(0,1))#z轴
        numx=np.squeeze(np.array(np.where(X1!=0)))#求非0值位置索引,
        numy=np.squeeze(np.array(np.where(X2!=0)))
        numz=np.squeeze(np.array(np.where(X3!=0)))
        # print(i,X[i,numx[0]:numx[-1],numy[0]:numy[-1],numz[0]:numz[-1]].shape)
        Xtemp=X[i,numx[0]:numx[-1],numy[0]:numy[-1],numz[0]:numz[-1]]
        Ytemp=Y[i,numx[0]:numx[-1],numy[0]:numy[-1],numz[0]:numz[-1]]
        L.append(Xtemp.shape)
        index.append([numx[0],shapex-numx[-1],numy[0],shapey-numy[-1],numz[0],shapez-numz[-1]])
        # shape_x=Xtemp.shape
        # if shape_x[2]>=128:
        #print(i,X[i,numx[0]:numx[-1],numy[0]:numy[-1],numz[0]:numz[-1]].shape)
        if resample!=None:###可以设置灵活调节是否采样，或者只是取全脑
            Ytemp=onehot(Ytemp,n_label)
            Ys=zoom(Ytemp,zoom=[resample[0]/Xtemp.shape[0],resample[1]/Xtemp.shape[1],resample[2]/Xtemp.shape[2],1],order=1)
            Ys=restore_labels(Ys,range(n_label))
            Xs=zoom(Xtemp,zoom=[resample[0]/Xtemp.shape[0],resample[1]/Xtemp.shape[1],resample[2]/Xtemp.shape[2]],order=1)###全写成shape[0]了
            #print(Xs.shape,Ys.shape)
            X_final.append(Xs)
            Y_final.append(Ys)
        else:
            X_final.append(Xtemp)
            Y_final.append(Ytemp)
    return X_final,Y_final,L,index###出于灵活考虑，输出均为列表，需要的话后续转为numpy








def pad_edge_sample(X,Y,L,n_label=None):###将crop_edge_sample的图像，一般只是针对标签，因为X可以使用原图像


    
    x0=[]
    y0=[]
    X=np.array(X)
    Y=np.array(Y)
    for i in range(X.shape[0]):
        Xs=zoom(X[i],zoom=[L[i][0]/X.shape[1],L[i][1]/X.shape[2],L[i][2]/X.shape[3]],order=1)
        if n_label==None:
            Ys=zoom(Y[i],zoom=[L[i][0]/X.shape[1],L[i][1]/X.shape[2],L[i][2]/X.shape[3]],order=1)
        else:
            Ys=onehot(Y[i],n_label)
            Ys=zoom(Ys,zoom=[L[i][0]/X.shape[1],L[i][1]/X.shape[2],L[i][2]/X.shape[3],1],order=1)
            Ys=restore_labels(Ys,range(n_label))
        x0.append(Xs)
        y0.append(Ys)
    return x0,y0


def Y_pad_sampe(Y,L,index,n_label=None):####仅仅还原Y的函数
    small = 0   
    Y=np.array(Y)
    y0=[]
    for i in range(Y.shape[0]):
        if n_label==None:
            Ys=zoom(Y[i],zoom=[L[i][0]/Y.shape[1],L[i][1]/Y.shape[2],L[i][2]/Y.shape[3]],order=1)
            Ys=np.pad(Ys, [[index[i][0], index[i][1]], [index[i][2], index[i][3]], [index[i][4], index[i][5]]], 'constant', constant_values=small)
        else:
            Ys=onehot(Y[i],n_label)
            Ys=zoom(Ys,zoom=[L[i][0]/Y.shape[1],L[i][1]/Y.shape[2],L[i][2]/Y.shape[3],1],order=1)
            Ys=np.pad(Ys, [[index[i][0], index[i][1]], [index[i][2], index[i][3]], [index[i][4], index[i][5]],[0,0]], 'constant', constant_values=small)

            Ys=restore_labels(Ys,range(n_label))
        y0.append(Ys)
    return y0  


def Y_pad_single(Y,L,index,n_label=None):####仅仅还原Y的函数,只针对单张图片，更方便
    small = 0   
    Y=np.array(Y)
    y0=[]
    
    if n_label==None:
        Ys=zoom(Y,zoom=[L[0]/Y.shape[0],L[1]/Y.shape[1],L[2]/Y.shape[2]],order=1)
        Ys=np.pad(Ys, [[index[0], index[1]], [index[2], index[3]], [index[4], index[5]]], 'constant', constant_values=small)
    else:
        Ys=zoom(Y,zoom=[L[0]/Y.shape[0],L[1]/Y.shape[1],L[2]/Y.shape[2],1],order=1)
        Ys=np.pad(Ys, [[index[0], index[1]], [index[2], index[3]], [index[4], index[5]],[0,0]], 'constant', constant_values=small)

        
    return Ys


# def fft_sample():
# def fftsample(data,ratio=4):
#     ##data=[256,256,256]
#     sample_shape=[data.shape[0]/ratio,data.shape[1]/ratio,data.shape[2]/ratio]
#     center=[data.shape[0]/2,data.shape[1]/2,data.shape[2]/2]
#     start=[center[0]-sample_shape[0]/2-1,center[1]-sample_shape[1]/2-1,center[2]-sample_shape[2]/2-1]
#     end=[center[0]+sample_shape[0]/2-1,center[1]+sample_shape[1]/2-1,center[2]+sample_shape[2]/2-1]
#     start=np.array(start,dtype=np.int16)
#     end=np.array(end,dtype=np.int16)
#     data3D=data
#     fft_data3D=np.fft.fftn(data3D)/(ratio**3)###注意采样要缩小频域值，否则重建回去的值可以达到几万,r*r*r
#     shift2center = np.fft.fftshift(fft_data3D)
#     # print(shift2center.shape)

#     crop_fft3D=shift2center[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
#     final_data3D=np.abs(np.fft.ifftn(crop_fft3D))
#     # print(final_data3D.shape)
#     return final_data3D

def fftsample(data,ratio=4,sample_pad=False):
    ##data=[256,256,256]
    sample_shape=[data.shape[0]/ratio,data.shape[1]/ratio,data.shape[2]/ratio]
    center=[data.shape[0]/2,data.shape[1]/2,data.shape[2]/2]
    start=[center[0]-sample_shape[0]/2-1,center[1]-sample_shape[1]/2-1,center[2]-sample_shape[2]/2-1]
    end=[center[0]+sample_shape[0]/2-1,center[1]+sample_shape[1]/2-1,center[2]+sample_shape[2]/2-1]
    start=np.array(start,dtype=np.int16)
    end=np.array(end,dtype=np.int16)
    data3D=data
    if sample_pad:
        fft_data3D=np.fft.fftn(data3D)
    else:

        fft_data3D=np.fft.fftn(data3D)/(ratio**3)###注意采样要缩小频域值，否则重建回去的值可以达到几万,r*r*r
    shift2center = np.fft.fftshift(fft_data3D)
    # print(shift2center.shape)

    crop_kspace=shift2center[start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    if sample_pad:
        print('crop_fft3D shape',crop_kspace.shape)
        crop_fft3D =crop_pad3D(crop_kspace,list(data.shape))
    final_data3D=np.abs(np.fft.ifftn(crop_fft3D))
    
    # print(final_data3D.shape)
    return final_data3D


def fft_guasswindow(data,sigma=15,inverse=False):##直接用方形截取会出现轮状伪影，这里的设置是改成使用高斯窗：
    shape = data.shape
    m, n, p = [(ss-1.)/2. for ss in shape]
    z, y, x = np.ogrid[-m:m+1, -n:n+1, -p:p+1]
    h = np.exp(-(x*x + y*y + z*z) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.max()
    # if sumh != 0:
    h /= sumh
    print('h',h.shape,data.shape,m, n, p)
    fft_data3D=np.fft.fftn(data)
    shift2center = np.fft.fftshift(fft_data3D)
    print('shift2center',shift2center.shape)
    if inverse:
        crop_fft3D = shift2center*(1-h)
    else:
        crop_fft3D = shift2center*h
    final_data3D=np.abs(np.fft.ifftn(crop_fft3D))
    return final_data3D,h




def patch_map(x,H=100,D=80,xl=16,yl=8):
    'H,D是单张图像的长宽,而xl,yl是合并map的图像数，即列是几张，行是几张'
    '此函数用于将一组2D图像合并为一张MAP，特别适合用于特征图'
    map = np.zeros([H*xl,D*yl])
    index =0
    for i in range(xl):
        for j in range(yl):
            map[i*H:(i+1)*H,j*D:(j+1)*D]=x[index]
            index +=1
    return map


def create_randm_mask(batch_size=2,mask_size=[96,96,96],patch=[4,4,4],prob=0.5,device =torch.device("cuda")):
    'prob设置掩码比例,此比例变为0'
    shape_size=[int(mask_size[0]/patch[0]),int(mask_size[1]/patch[1]),int(mask_size[1]/patch[1])]
    #print(shape_size)
    mask =np.random.uniform(size=list(shape_size))
    mask[mask<prob]=0
    mask[mask>prob]=1
    mask_final=np.zeros(mask_size)
    for i in range(shape_size[0]):
        for j in range(shape_size[1]):
            for k in range(shape_size[2]):
                mask_final[patch[0]*i:patch[1]*(i+1),patch[1]*j:patch[1]*(j+1),patch[2]*k:patch[2]*(k+1)]=mask[i,j,k]

    save_nifit(mask_final,'/home/yunzhixu/Data/ASL+PET_XuYunZhi/mask_final.nii.gz')

    ##to tenosr
    mask_final = mask_final[np.newaxis,:,:,:]
    mask_tensor = torch.zeros([batch_size,1,mask_size[0],mask_size[1],mask_size[2]])
    for i in range(batch_size):
        mask_tensor[i]=torch.tensor(mask_final)
    
    mask_tensor = mask_tensor.to(device)

    return mask_tensor




def div0(a, b):
    if b == 0:
        c = np.zeros_like(a)
    else:
        c = a / b
    return c
def normalize_mri(image,return_stats=False):
    # Z-score 标准化
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = div0((image - mean), std)

    # 线性映射到 [-1, 1]
    min_norm = np.min(normalized_image)
    ptp_norm = np.ptp(normalized_image)
    normalized_image = 2 * (normalized_image - np.min(normalized_image)) / np.ptp(normalized_image) - 1
    if return_stats:
        return normalized_image, mean, std,min_norm, ptp_norm
    else:
        return normalized_image

def normalize_mri_array(tmp,num_dim=True):
    '对于一组数据进行z_score_norm'
    if num_dim:
        data =[]
        for i in range(tmp.shape[0]):
            data_tmp = normalize_mri(tmp[i])
            data.append(data_tmp)
        data =np.array(data)
    else:
        data=normalize_mri(data)

    return data


def z_score_syn(image,pred,windows = [-1,4]):
    '''
    同步版zsocre归一化，因为之前的mri都是最大值最小值归一化，但是对于有的预测图像，其值的极端范围比较大，因此可以固定值
    '''
    mi = np.mean(image)
    mp =np.mean(pred)

    pred = (pred-mp)/np.std(pred)
    image = (image-mi)/np.std(image)
    #print(np.mean(pred),np.mean(image),np.std(pred),np.std(image))

    start = windows[0]
    lens = windows[1] - windows[0]

    pred = (pred-start)/lens
    image = (image-start)/lens
    image[image>1] = 1
    pred[pred>1] = 1
    image[image<0] = 0
    pred[pred<0] = 0
    return image,pred

def z_score_syn_tensor(image, pred, windows=(-1, 4)):
    """
    Torch版同步Z-score归一化
    :param image: 输入图像 (torch.Tensor)
    :param pred: 预测图像 (torch.Tensor)
    :param windows: 固定归一化范围 (tuple)
    :return: 归一化后的图像和预测图像
    """
    mi = torch.mean(image)
    mp = torch.mean(pred)

    pred = (pred - mp) / torch.std(pred)
    image = (image - mi) / torch.std(image)

    start = windows[0]
    lens = windows[1] - windows[0]

    pred = (pred - start) / lens
    image = (image - start) / lens

    pred = torch.clamp(pred, 0, 1)
    image = torch.clamp(image, 0, 1)

    return image, pred


def norm_MRI(data):
    data = np.array(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data

##seg
def plot_seg_color(data,labels_seg):
    C=[0.5,0,0]
    data_0=norm_MRI(data)
    r=data_0.copy()
    g=data_0.copy()
    b=data_0.copy()
    r[labels_seg==1]=C[0]
    # g[labels_seg==1]=C[1]
    # b[labels_seg==1]=C[2]
    g2=np.zeros([data.shape[0],data.shape[1],data.shape[2],3])
    g2[...,0]=r
    g2[...,1]=g
    g2[...,2]=b
    print(g2.shape)
    g2=np.array(g2*255,np.uint16)

    # plt.imshow(g2)
    return g2
def restore_labels(x, labels):
    tmp = np.squeeze(np.argmax(x, -1)).astype(np.int8)
    y = np.zeros(tmp.shape, np.int8)
    n_labels = len(labels)
    for label_index in range(n_labels):
        y[tmp == label_index] = labels[label_index]
    return y

def direction_set(img,direction):
    '''
    纠正朝向
    '''
    direction = np.array(direction).reshape(3,3)
    direction = np.round(direction,0)
    if np.abs(np.abs(direction[0,0])-1)<0.01:
        if np.abs(np.abs(direction[1,1])-1)<0.01:
            img = img
        elif np.abs(np.abs(direction[2,1])-1)<0.01:
            img = np.transpose(img,(0,2,1))
        else:
            print('error')
    elif np.abs(np.abs(direction[1,0])-1)<0.01:
        if np.abs(np.abs(direction[0,1])-1)<0.01:
            img = np.transpose(img,(1,0,2))
        elif np.abs(np.abs(direction[2,1])-1)<0.01:
            img = np.transpose(img,(1,2,0))    
        else:
            print('error')  
    elif np.abs(np.abs(direction[2,0])-1)<0.01:
        if np.abs(np.abs(direction[0,1])-1)<0.01:
            img = np.transpose(img,(2,0,1))
        elif np.abs(np.abs(direction[1,1])-1)<0.01:
            img = np.transpose(img,(2,1,0))

        else:
            print('error')
    else:
        print('error')
    return img


# setup_seed(42)

def display_center_slices(volume, title="Center Slices", save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    center_slices = [s // 2 for s in volume.shape]

    axes[0].imshow(volume[center_slices[0], :, :], cmap="gray")
    axes[0].set_title("Axial Slice")
    axes[0].axis("off")

    axes[1].imshow(volume[:, center_slices[1], :], cmap="gray")
    axes[1].set_title("Coronal Slice")
    axes[1].axis("off")

    axes[2].imshow(volume[:, :, center_slices[2]], cmap="gray")
    axes[2].set_title("Sagittal Slice")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


from skimage.exposure import match_histograms

def align_contrast(target_img, reference_img_path):
    """
    将 target_img (eT1w) 的直方图匹配到 reference_img (ADNI T1w)
    """
    # 随机加载一个 ADNI 的数据作为参考
    ref_img = load_nifit(reference_img_path) # 你需要实现简单的加载
    
    # 计算匹配
    matched = match_histograms(target_img, ref_img, channel_axis=None)
    return matched