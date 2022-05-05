import torch
import torch.nn as nn
from torchvision import transforms, models
class ROILAnet(nn.Module):
    def __init__(self, h=56, w=56, L=18):
        super(ROILAnet, self).__init__()
        self.h = h
        self.w = w
        self.L = L
        vgg16 = models.vgg16(pretrained=False) # load vgg16 with pretrained weights
        vgg16 = vgg16.features # only get feature block
        vgg16 = vgg16[0:18] # cut off after first three conv-blocks
        vgg16[-1] = torch.nn.LocalResponseNorm(512*2, 1e-6, 1, 0.5) #local response normalisation´
        self.featureExtractionCNN = vgg16
        self.featureExtractionCNN.requires_grads=False
        # Regression network
        self.regressionNet = nn.Sequential(
            nn.Linear(int(self.h/8) * int(self.w/8) * 256, 512),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, self.L)
        )  
        self.regressionNet.requires_grads=False
    
    def forward(self,I_resized):
        # Pass to feature extraction CNN
        feat = self.featureExtractionCNN(I_resized)
        feat  = feat.view(-1, int(self.h/8) * int(self.w/8) * 256)
        # Pass to regression network
        theta = self.regressionNet(feat)
        return theta