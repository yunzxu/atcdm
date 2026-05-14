
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tool.image_process import save_nifit

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

    
setup_seed(42)
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    v=v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))



class SRDiffusionTrainer(nn.Module):
    def __init__(self, model,LR_EN, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.LR_EN =LR_EN
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def sample(self,x_0,t):
        t = t*torch.ones(size=(x_0.shape[0], ))
        device=x_0.device
        t= torch.tensor(t).type(torch.int64).to(device)
        print(t)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_t

    def forward(self, x_0,x_e):##把低dose图像作为噪声
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)


        x_lr = self.LR_EN(x_e) ##输入隐变量
        x_t = x_t + x_lr
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class SRDiffusionSampler(nn.Module):
    def __init__(self, model,LR_EN, beta_1, beta_T, T,infer_T=None):
        super().__init__()

        self.model = model
        self.LR_EN =LR_EN
        self.T = T
        if infer_T ==None:
            self.infer_T =T
        else:
            self.infer_T = infer_T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, x_e, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        x_lr = self.LR_EN(x_e)
        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        """
        Algorithm 2.
        """
        
        x_t = x_T[:,0:1,...]
        x_e = x_T[:,1:2,...]
        infer_num = 0
        print('T',self.T,'infer_T',self.infer_T)
        for time_step in reversed(range(self.infer_T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t,x_e=x_e ,t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            infer_num +=1
            # if infer_num==self.infer_T:
            #     break
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   
    





class SR3DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model

        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def sample(self,x_0,t):
        t = t*torch.ones(size=(x_0.shape[0], ))
        device=x_0.device
        t= torch.tensor(t).type(torch.int64).to(device)
        print(t)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_t

    def forward(self, x_0,x_e):##把低dose图像作为噪声
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)


    
        x_t = torch.cat([x_t,x_e],dim=1)


        ### 对于多连续切片，或许应该相邻额嵌入
        # x_cat = torch.cat([x_t[:,0:1],x_e[:,0:1]],dim=1)
        # if x_0.shape[1]>1:
        #     for i in  range(1,x_0.shape[1]):
        #         tmp = torch.cat([x_t[:,i:i+1],x_e[:,i:i+1]],dim=1)
        #         x_cat = torch.cat([x_cat,tmp],dim=1)
        # x_t = x_cat
        # save_nifit(x_t.cpu().numpy()[0],'/data/yunzhixu/medical_fusion/ASL_super_res/predict/train_diffusion/train_SR3DDPM_normadni_GMASL_LRHR_2Dslice_T1000_slice5_newcat/xt.nii.gz')
          
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class SR3DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T,infer_T=None):
        super().__init__()

        self.model = model
        self.T = T
        if infer_T ==None:
            self.infer_T =T
        else:
            self.infer_T = infer_T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
    


    def sample(self,x_0,t):
        t = t*torch.ones(size=(x_0.shape[0], ))
        device=x_0.device
        t= torch.tensor(t).type(torch.int64).to(device)
        # print(t)
        noise = torch.randn_like(x_0)
        # print('noise',noise.shape)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        return x_t
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, x_e, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        x_te = torch.cat([x_t,x_e],dim=1)
        eps = self.model(x_te, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T,x_e=None):
        """
        Algorithm 2.
        """

        if x_e == None:
            x_t = x_T[:,0:1,...]
            x_e = x_T[:,1:,...]
        else:
            x_t = x_T
        
    
        infer_num = 0
        print('T',self.T,'infer_T',self.infer_T)
        for time_step in reversed(range(self.infer_T)):
            # print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t,x_e=x_e ,t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise

            ##SR3_guide:
            # x_t = ((self.infer_T-time_step)/self.infer_T)*x_t  +(time_step/self.infer_T)*self.sample(x_e,time_step)
            
            #0.9*x_t + 0.1*self.guide_sample(x_guide,time_step)

            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            infer_num +=1
            # if infer_num==self.infer_T:
            #     break
        x_0 = x_t
        # return torch.clip(x_0, -1, 1)   
        return  x_0
    


class SR3DiffusionSampler2D(nn.Module): ##用于2D 多通道的输入

    def __init__(self, model, beta_1, beta_T, T,infer_T=None):
        super().__init__()

        self.model = model
        self.T = T
        if infer_T ==None:
            self.infer_T =T
        else:
            self.infer_T = infer_T

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, x_e, t):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        x_te = torch.cat([x_t,x_e],dim=1)

        ###新修改，x_t和x_e交替cat
        # x_cat = torch.cat([x_t[:,0:1],x_e[:,0:1]],dim=1)
        # if x_e.shape[1]>1:
        #     for i in  range(1,x_e.shape[1]):
        #         tmp = torch.cat([x_t[:,i:i+1],x_e[:,i:i+1]],dim=1)
        #         x_cat = torch.cat([x_cat,tmp],dim=1)
        # x_te = x_cat
        # print('x_te',x_te.shape)



        eps = self.model(x_te, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_t,x_e):
        """
        Algorithm 2.
        """
        
        x_t = x_t
        x_e = x_e
        infer_num = 0
        # print('T',self.T,'infer_T',self.infer_T)
        for time_step in reversed(range(self.infer_T)):
            # print(time_step)
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t,x_e=x_e ,t=t)
            # no noise when t == 0
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            infer_num +=1
            # if infer_num==self.infer_T:
            #     break
        x_0 = x_t
        return torch.clip(x_0, -1, 1)  



# from model import UNet
# from torchsummary import summary
# if __name__ == "__main__":
#     net_model=UNet(
#         T=1000, ch=32, ch_mult=[1, 2, 2, 2], attn=[1],
#         num_res_blocks=2, dropout=0.1).cuda()


#     trainer = GaussianDiffusionTrainer(
#     net_model, 1e-4, 0.02,1000).cuda()


#     x = torch.randn(2, 1, 96, 96,96).cuda()
#     loss=trainer(x)
#     print(x.shape)
#     summary(trainer,[1,96,96,96])
#     print(loss.shape)


        # sample_eval = GaussianDiffusionSampler(net_model, 1e-4, 0.02,100)

        # y_s=sample_eval(x)
        # print(y_s.shape)

