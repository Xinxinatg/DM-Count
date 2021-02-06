from PIL import Image
import requests
#import matplotlib.pyplot as plt
#%config InlineBackend.figure_format = 'retina'

import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
from .compressed_transformer import Transformer
import math
#import torch_xla
#import torch_xla.core.xla_model as xm
#dev=to.class TR_CC(nn.Module):
dev='cuda'
class TR_CC(nn.Module):
    def __init__(self, hidden_dim=24, nheads=4,
                 num_encoder_layers=6,load_weights=False):
        super().__init__()

        # create ResNet-50 backbone
        self.seen = 0
        # create conversion layer
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = make_layers(self.frontend_feat)
        self.conv = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        #build_transformer(args)
        self.transformer = Transformer(d_model=hidden_dim,nhead=nheads)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.output_layer=nn.Conv2d(hidden_dim, 1, kernel_size=1)
    
        # output positional encodings (object queries)

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(400, hidden_dim // 2,requires_grad=True, device=dev))
        self.col_embed = nn.Parameter(torch.rand(400, hidden_dim // 2,requires_grad=True, device=dev))
        self.output_norm=nn.ReLU(inplace=True)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        #self._initialize_weights()
      #  self.row_embed = nn.Parameter(torch.rand(100, hidden_dim // 2))
      #  self.col_embed = nn.Parameter(torch.rand(100, hidden_dim // 2))
    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x=self.frontend(inputs)
        src = self.conv(x)
        bs, c, h, w = src.shape
    #    print(bs,c,h,w)
        #src = src.flatten(2).permute(2, 0, 1)
        #pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        pos = torch.cat([
            self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
            self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
        # propagate through the transformer
        h = self.transformer(0.1 * src,pos)
        densitym=self.output_layer(h)
        densitym=self.output_norm(densitym)
        B, C, H, W = densitym.size()
        densitym_sum = densitym.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        densitym_normed = densitym / (densitym_sum + 1e-6)
        return densitym, densitym_normed

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fansum = m.weight.size(1) + m.weight.size(0)
                scale = 1. / max(1., float(fansum) / 2.)
                stdv = math.sqrt(3. * scale)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)       
