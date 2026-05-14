import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import math
import os
from torchsummary import summary

class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #print("value_layer",value_layer.shape)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #print("attention_scores",attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        #print("attention_probs",attention_probs.shape)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #print("context_layer",context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        #print("new_context_layer_shape",new_context_layer_shape)
        context_layer = context_layer.view(*new_context_layer_shape)
        #print("context_layer",context_layer.shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        #print("x.flatten",x.shape)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.attn = SelfAttention(num_heads, embed_dim, dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x

        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class UNETR(nn.Module):
    def __init__(self, img_shape=(96, 96, 96), input_dim=1, output_dim=8, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]

        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SingleConv3DBlock(64, output_dim, 1)
            )
        # self.outlayer = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        # output=self.outlayer(output)
        return output







class conv_block_nested_3D(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch,bn=False,use_res=False):
        super(conv_block_nested_3D, self).__init__()
        self.use_res=use_res
        self.bn=bn
        self.activation = nn.LeakyReLU(inplace=True)#nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.res= nn.Conv3d(in_ch, out_ch, kernel_size=1,bias=True)
        self.bn3 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        input=x
        #input=self.conv3(x)###残差连接
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)
    
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        if self.use_res:
            res=self.res(input)
            res=self.bn3(res)
            x=x+res
        output = self.activation(x)#+input



        return output


class UNetPlusPlus_nest3_3d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, n1=16, deep_supervision=False):
        super(UNetPlusPlus_nest3_3d, self).__init__()
        print('nest3 used')

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv1_1 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv1_2 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_1 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)





        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3])

        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2])

        self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1])

        self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    # def forward(self, x):
    #     print(x.shape)
    #     x0_0 = self.conv0_0(x)
    #     x1_0 = self.conv1_0(self.pool(x0_0))
    #     x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

    #     x2_0 = self.conv2_0(self.pool(x1_0))
    #     x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
    #     x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

    #     x3_0 = self.conv3_0(self.pool(x2_0))
    #     x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
    #     x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
    #     x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

    #     if self.deep_supervision:
    #         output1 = self.final1(x0_1)
    #         output2 = self.final2(x0_2)
    #         output3 = self.final3(x0_3)
    #         return [output1, output2, output3]

    #     else:
    #         output = self.final(x0_3)
    #         return output

    def forward(self, x):
        # print(x.shape)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Deconv1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Deconv2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Deconv1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Deconv3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Deconv2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Deconv1_2(x1_2)], 1))

        # print('x1_0',x1_0.shape)
        # print('x2_0',x2_0.shape)
        # print('x3_0',x3_0.shape)
        # print('x2_1',x2_1.shape)
        # print('x1_2',x1_2.shape)
        # print('x1_1',x1_1.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output

class UNet_nest3_3d(nn.Module):###在unet++基础上的unet
        
    def __init__(self, in_channels=1, out_channels=1, n1=64, deep_supervision=False,bn=True,use_res=True):
        super(UNet_nest3_3d, self).__init__()
        print('nest3 used')

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 *16]
        self.deep_supervision = deep_supervision
     

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv4_0 =nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2, padding=0, output_padding=0)

        # self.Deconv1_1 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.Deconv1_2 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.Deconv2_1 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)





        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0],bn,use_res)
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1],bn,use_res)
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2],bn,use_res)
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3],bn,use_res)
        self.conv4_0 = conv_block_nested_3D(filters[3], filters[4], filters[4],bn,use_res)
        



        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0],bn,use_res)
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1],bn,use_res)
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2],bn,use_res)
        self.conv3_1 = conv_block_nested_3D(filters[3] + filters[4], filters[3], filters[3],bn,use_res)


        # self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0],bn)
        # self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1],bn)

        # self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0],bn)

        if self.deep_supervision:
            self.final1 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
    
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        #print('x1-4',x0_0.shape,x1_0.shape,x2_0.shape,x3_0.shape,x4_0.shape)
        # print('x1_0',x1_0.shape)
        # print('x2_0',x2_0.shape)
        #print('self.Deconv3_0(x3_0)',self.Deconv3_0(x3_0).shape)
        x3_1 =self.conv3_1(torch.cat([x3_0, self.Deconv4_0(x4_0)], 1))
        #print('x3_1',x3_1.shape)


        x2_1 = self.conv2_1(torch.cat([x2_0, self.Deconv3_0(x3_1)], 1))
        # print('X2_1',x2_1.shape)
        # print('self.Deconv2_1(x2_1)',self.Deconv2_1(x2_1).shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Deconv2_0(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Deconv1_0(x1_1)], 1))

        # if self.deep_supervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     return [output1, output2, output3]

        # else:
        output = self.final(x0_1)
        return output




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels 
        # if in_channels > out_channels:
        #     channels = in_channels // 2

        layers = [
            # in_channels：输入通道数
            # channels：输出通道数
            # kernel_size：卷积核大小
            # stride：步长
            # padding：边缘填充
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal: # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=False):
        super(UpSampling, self).__init__()
        if bilinear:
            # 采用双线性插值的方法进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 采用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batch_normal)

    # inputs1：上采样的数据（对应图中黄色箭头传来的数据）
    # inputs2：特征融合的数据（对应图中绿色箭头传来的数据）
    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)

        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        print('outputs',outputs.shape)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=False,ngf=16):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear


        self.inputs = DoubleConv(in_channels, ngf, self.batch_normal)
        self.down_1 = DownSampling(ngf, ngf*2, self.batch_normal)
        self.down_2 = DownSampling(ngf*2, ngf*4, self.batch_normal)
        self.down_3 = DownSampling(ngf*4, ngf*8, self.batch_normal)

        self.up_1 = UpSampling(ngf*8, ngf*4, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(ngf*4, ngf*2, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(ngf*2, ngf, self.batch_normal, self.bilinear)
        self.outputs = LastConv(ngf, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)
        print('x1',x1.shape)
        x2 = self.down_1(x1)
        print('x2',x2.shape)
        x3 = self.down_2(x2)
        print('x3',x3.shape)
        x4 = self.down_3(x3)
        print('x4',x4.shape)

        # up部分
        print('x4,x3',x4.shape,x3.shape)
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)

        return x



from torchsummary import summary

if __name__ == "__main__":
    model=UNet_nest3_3d().cuda()
    summary(model,[1,96,96,96])
