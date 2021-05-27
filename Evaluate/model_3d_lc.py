import math
import numpy as np
import sys
sys.path.append('../Backbone')
from Backbone.select_backbone import select_resnet
from Backbone.transformer_encoder import TransformerEncoders
from Backbone.convrnn import ConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F

class LC(nn.Module):
    def __init__(self, sample_size, num_seq, network='resnet18', dropout=0.5, num_class=101):
        super(LC, self).__init__()
        torch.cuda.manual_seed(666)
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.num_class = num_class
        self.last_size = int(math.ceil(sample_size / 32))  # 4
        print('=> Using CPCTrans + FC model ')

        print('=> Use backbone %s!' % network)

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1  # param for Transformer
        self.param['num_heads'] = 2  # param for Transformer
        self.param['num_cells'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        print('=> using TransformerEncoder ')
        self.transEnc = TransformerEncoders(d_model=self.param['feature_size'],
                                            N=self.param['num_layers'],
                                            h=self.param['num_heads'])
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_cells'])
        self._initialize_weights(self.agg)
        self._initialize_weights(self.transEnc)

        self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
        self.final_bn.weight.data.fill_(1)
        self.final_bn.bias.data.zero_()

        self.final_fc = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(self.param['feature_size'], self.num_class))
        self._initialize_weights(self.final_fc)

    def forward(self, block):
        (B, N, C, H, W) = block.shape
        block = block.view(B * N, C, H, W)
        feature = self.backbone(block)  # feature of shape (B*N, feature_size(256), 4, 4)
        del block

        feature = F.relu(feature)
        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size,
                                       self.last_size)
        ### feed to transformer encoder, then predict future ###
        feature_inf_all = feature_inf_all.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, self.param['feature_size'])
        seq_trans = self.transEnc(feature_inf_all)
        seq_trans = seq_trans.view(B, N, self.last_size, self.last_size, self.param['feature_size'])
        seq_trans = seq_trans.permute(0, 1, 4, 2, 3).contiguous()
        seq_trans = F.relu(seq_trans)
        del feature, feature_inf_all

        context, _ = self.agg(seq_trans)
        context = context[:, -1, :].unsqueeze(1)
        context = F.avg_pool3d(context, (1, self.last_size, self.last_size), stride=1).squeeze(-1).squeeze(-1)
        del seq_trans

        context = self.final_bn(context.transpose(-1, -2)).transpose(-1, -2)
        output = self.final_fc(context).view(B, -1, self.num_class)

        return output

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)


