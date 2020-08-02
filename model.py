# The detail shows in below supplementaly.
# https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Lee_Reference-Based_Sketch_Image_CVPR_2020_supplemental.pdf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=16, d_channel=448, channel_1x1=256):
        super(Generator, self).__init__()

        self.enc_r = Encoder(in_channel=3, conv_dim=conv_dim)
        self.enc_s = Encoder(in_channel=1, conv_dim=conv_dim)
        
        self.scft = SCFT(d_channel=d_channel)
        
        self.conv1x1 = spectral_norm(nn.Conv2d(d_channel, channel_1x1, kernel_size=1, stride=1, padding=0))
        
        self.resblocks = nn.Sequential(
            ResidualBlock(channel_1x1, channel_1x1),
            ResidualBlock(channel_1x1, channel_1x1),
            ResidualBlock(channel_1x1, channel_1x1),
            ResidualBlock(channel_1x1, channel_1x1),
        )
        
        
        self.decoder = Decoder(in_channel=channel_1x1)
        self.activation = nn.Tanh()
    
    def forward(self, I_r, I_s, IsGTrain=False):
        v_r = self.enc_r(I_r)
        v_s, I_s_f1_9 = self.enc_s(I_s)
        
        f_scft, L_tr = self.scft(v_r, v_s)
        
        f_encoded = self.conv1x1(f_scft)
        
        f_encoded = self.resblocks(f_encoded) + f_encoded # [1,512,32,32]

        f_out = self.decoder(f_encoded, I_s_f1_9)
        
        if IsGTrain:
            return self.activation(f_out), L_tr
        
        return self.activation(f_out)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1)))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=kernel_size, bias=False))
        
    def forward(self, x):
        h = self.main(x)
        out = self.conv1(h)
        return out.squeeze()


class Encoder(nn.Module):
    """Encoder network."""
    def __init__(self, in_channel=1, conv_dim=16):
        super(Encoder, self).__init__()
                
        self.conv1 = self.conv_block(in_channel, conv_dim, K=3, S=1, P=1)
        self.conv2 = self.conv_block(conv_dim, conv_dim, K=3, S=1, P=1)
        self.conv3 = self.conv_block(conv_dim, conv_dim*2, K=3, S=2, P=1)
        self.conv4 = self.conv_block(conv_dim*2, conv_dim*2, K=3, S=1, P=1)
        self.conv5 = self.conv_block(conv_dim*2, conv_dim*4, K=3, S=1, P=1)
        self.conv6 = self.conv_block(conv_dim*4, conv_dim*4, K=3, S=1, P=1)
        self.conv7 = self.conv_block(conv_dim*4, conv_dim*8, K=3, S=2, P=1)
        self.conv8 = self.conv_block(conv_dim*8, conv_dim*8, K=3, S=1, P=1)
        self.conv9 = self.conv_block(conv_dim*8, conv_dim*16, K=3, S=2, P=1)
        self.conv10 = self.conv_block(conv_dim*16, conv_dim*16, K=3, S=1, P=1)
        
        if in_channel == 3:
            self.mode = 'E_r'
        elif in_channel == 1:
            self.mode = 'E_s'
        else:
            raise NotImplementedError

    def conv_block(self, C_in, C_out, K=3, S=1, P=1):
        return nn.Sequential(
            # nn.Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P),
            spectral_norm(nn.Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P)),
            nn.InstanceNorm2d(C_out, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
    def forward(self, x): # x: [1,3,256,256]
        bs = x.size(0)
        
        f1 = self.conv1(x) # [1, 16, 256, 256]
        f2 = self.conv2(f1) # [1, 16, 256, 256]
        f3 = self.conv3(f2) # [1, 32, 128, 128]
        f4 = self.conv4(f3) # [1, 32, 128, 128]
        f5 = self.conv5(f4) # [1, 64, 128, 128]
        f6 = self.conv6(f5) # [1, 64, 128, 128]
        f7 = self.conv7(f6) # [1, 128, 64, 64]
        f8 = self.conv8(f7) # [1, 128, 64, 64]
        f9 = self.conv9(f8) # [1, 256, 32, 32]
        f10 = self.conv10(f9) # [1, 256, 32, 32]
        
        fs = (f1,f2,f3,f4,f5,f6,f7,f8,f9,f10)
        
        # f1 = F.interpolate(f1, scale_factor=f10.size(2)/f1.size(2), mode='nearest', recompute_scale_factor=True) # [1, 16, 32, 32]
        # f2 = F.interpolate(f2, scale_factor=f10.size(2)/f2.size(2), mode='nearest', recompute_scale_factor=True) # [1, 16, 32, 32]
        # f3 = F.interpolate(f3, scale_factor=f10.size(2)/f3.size(2), mode='nearest', recompute_scale_factor=True) # [1, 32, 32, 32]
        # f4 = F.interpolate(f4, scale_factor=f10.size(2)/f4.size(2), mode='nearest', recompute_scale_factor=True) # [1, 32, 32, 32]
        # f5 = F.interpolate(f5, scale_factor=f10.size(2)/f5.size(2), mode='nearest', recompute_scale_factor=True) # [1, 64, 32, 32]
        # f6 = F.interpolate(f6, scale_factor=f10.size(2)/f6.size(2), mode='nearest', recompute_scale_factor=True) # [1, 64, 32, 32]
        # f7 = F.interpolate(f7, scale_factor=f10.size(2)/f7.size(2), mode='nearest', recompute_scale_factor=True) # [1, 128, 32, 32]
        # f8 = F.interpolate(f8, scale_factor=f10.size(2)/f8.size(2), mode='nearest', recompute_scale_factor=True) # [1, 128, 32, 32]
        # f9 = F.interpolate(f9, scale_factor=f10.size(2)/f9.size(2), mode='nearest', recompute_scale_factor=True) # [1, 256, 32, 32]
        
        f1 = F.interpolate(f1, scale_factor=f10.size(2)/f1.size(2), mode='nearest') # [1, 16, 32, 32]
        f2 = F.interpolate(f2, scale_factor=f10.size(2)/f2.size(2), mode='nearest') # [1, 16, 32, 32]
        f3 = F.interpolate(f3, scale_factor=f10.size(2)/f3.size(2), mode='nearest') # [1, 32, 32, 32]
        f4 = F.interpolate(f4, scale_factor=f10.size(2)/f4.size(2), mode='nearest') # [1, 32, 32, 32]
        f5 = F.interpolate(f5, scale_factor=f10.size(2)/f5.size(2), mode='nearest') # [1, 64, 32, 32]
        f6 = F.interpolate(f6, scale_factor=f10.size(2)/f6.size(2), mode='nearest') # [1, 64, 32, 32]
        f7 = F.interpolate(f7, scale_factor=f10.size(2)/f7.size(2), mode='nearest') # [1, 128, 32, 32]
        f8 = F.interpolate(f8, scale_factor=f10.size(2)/f8.size(2), mode='nearest') # [1, 128, 32, 32]
        f9 = F.interpolate(f9, scale_factor=f10.size(2)/f9.size(2), mode='nearest') # [1, 256, 32, 32]
        
        V = torch.cat([f6,f8,f10], dim=1) # Eq.(1) # 64+128+256=448, [1, 448, 32, 32]
        V_bar = V.view(bs, V.size(1), -1) # [1, 448, 1024]
        
        
        if self.mode == 'E_r':
            return V_bar
        elif self.mode == 'E_s':
            return V_bar, fs
        
class SCFT(nn.Module):
    def __init__(self, d_channel=448):
        super(SCFT, self).__init__()
        
        self.W_v = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
        self.W_k = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
        self.W_q = nn.Parameter(torch.randn(d_channel, d_channel)) # [448, 448]
        self.coef = d_channel ** .5
        
        self.gamma = 12.
    
    def forward(self, V_r, V_s):
        
        wq_vs = torch.matmul(self.W_q, V_s) # [1, 448, 1024]
        wk_vr = torch.matmul(self.W_k, V_r).permute(0, 2, 1) # [1, 448, 1024]
        alpha = F.softmax(torch.matmul(wq_vs, wk_vr) / self.coef, dim=-1) # Eq.(2)
        
        wv_vr = torch.matmul(self.W_v, V_r)
        v_asta = torch.matmul(alpha, wv_vr) # [1, 448, 1024] # Eq.(3) 
        
        c_i = V_s + v_asta # [1, 448, 1024] # Eq.(4)
        
        bs,c,hw = c_i.size()
        spatial_c_i = torch.reshape(c_i.unsqueeze(-1), (bs,c,int(hw**0.5), int(hw**0.5))) #  [1, 448, 32, 32]
        
        # Similarity-Based Triplet Loss
        a = wk_vr[0, :, :].detach().clone()
        b = wk_vr[1:, :, :].detach().clone()
        wk_vr_neg = torch.cat((b, a.unsqueeze(0)))
        alpha_negative = F.softmax(torch.matmul(wq_vs, wk_vr_neg) / self.coef, dim=-1)
        v_negative = torch.matmul(alpha_negative, wv_vr)
        
        L_tr = F.relu(-v_asta + v_negative + self.gamma).mean()
        
        return spatial_c_i, L_tr
        
        
class Decoder(nn.Module):
    def __init__(self, in_channel=256, out_channel=3):
        super(Decoder, self).__init__()
        
        self.deconv1 = self.conv_block(in_channel, in_channel)
        self.deconv2 = self.conv_block(in_channel, in_channel//2, upsample=True)
        in_channel //= 2
        self.deconv3 = self.conv_block(in_channel, in_channel)
        self.deconv4 = self.conv_block(in_channel, in_channel//2, upsample=True)
        in_channel //= 2
        self.deconv5 = self.conv_block(in_channel, in_channel)
        self.deconv6 = self.conv_block(in_channel, in_channel//2)
        in_channel //= 2
        self.deconv7 = self.conv_block(in_channel, in_channel)
        self.deconv8 = self.conv_block(in_channel, in_channel//2, upsample=True)
        in_channel //= 2
        self.deconv9 = self.conv_block(in_channel, in_channel)
        self.deconv10 = self.conv_block(in_channel, in_channel)
        
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(in_channel, affine=True),
            nn.LeakyReLU(0.2),
            # nn.Conv2d(in_channel, 3, 1, 1, 0)
            spectral_norm(nn.Conv2d(in_channel, 3, 1, 1, 0))
            )
        
        
    def conv_block(self, C_in, C_out, K=3, S=1, P=1, upsample=False):
        layers = [spectral_norm(nn.Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=P))]
        if upsample:
            layers += [nn.Upsample(scale_factor=2)]
        layers += [
            nn.InstanceNorm2d(C_out, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]
        return nn.Sequential(*layers)
        
    def forward(self, f_encoded, fs):
        f1,f2,f3,f4,f5,f6,f7,f8,f9,f10 = fs
        
        d1 = self.deconv1(f_encoded) # [1, 256, 32, 32]
        
        d1 = d1 + f9 # [1, 256, 32, 32]        
        d2 = self.deconv2(d1) # [1, 128, 64, 64]
        
        d2 = d2 + f8 # [1, 128, 64, 64]
        d3 = self.deconv3(d2) # [1, 128, 64, 64]
        
        d3 = d3 + f7 # [1, 128, 64, 64]
        d4 = self.deconv4(d3) # [1, 64, 64, 64]
        
        d4 = d4 + f6 # [1, 64, 128, 128]
        d5 = self.deconv5(d4) # [1, 64, 64, 64]
        
        d5 = d5 + f5 # [1, 64, 128, 128]
        d6 = self.deconv6(d5) # [1, 32, 128, 128]
        
        d6 = d6 + f4 # [1, 32, 128, 128]
        d7 = self.deconv7(d6) # [1, 32, 128, 128]
        
        d7 = d7 + f3 # [1, 32, 128, 128]
        d8 = self.deconv8(d7) # [1, 16, 256, 256]
        
        d8 = d8 + f2 # [1, 16, 256, 256]
        d9 = self.deconv9(d8) # [1, 16, 256, 256]
        
        d9 = d9 + f1 # [1, 16, 256, 256]
        d10 = self.deconv10(d9) # [1, 16, 256, 256]
        
        out = self.to_rgb(d10)

        return out
        

if __name__ == '__main__':
    I_s = torch.randn(4,1,256,256)
    I_r = torch.randn(4,3,256,256)
    
    G = Generator()
    G(I_r, I_s)