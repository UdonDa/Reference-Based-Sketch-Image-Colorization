import torch
import torch.nn as nn
from vgg import VGG16FeatureExtractor


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = VGG16FeatureExtractor()

    def forward(self, real, fake):
        loss_dict = {}

        feat_real = self.extractor(real)
        feat_real = self.extractor(fake)

        L_prec = 0.
        L_style = 0.
        for i in range(len(feat_real)):
            L_prec += self.l1(feat_real[i], feat_real[i])
            L_style += self.l1(gram_matrix(feat_real[i]), gram_matrix(feat_real[i]))

        L_prec = L_prec.mean()
        L_style = L_style.mean()
        

        
        return L_prec, L_style
    
    
if __name__ == '__main__':
    loss = VGGLoss()
    
    x = torch.randn(1,3,256,256)
    
    l1, l2 = loss(x,x)
    
    print(l1, l2)