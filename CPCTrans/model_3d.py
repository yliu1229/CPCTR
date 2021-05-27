import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../backbone')
from Backbone.select_backbone import select_resnet
from Backbone.transformer_encoder import TransformerEncoders
from Backbone.convrnn import ConvGRU


class CPC_Trans(nn.Module):
    '''CPC with Transformer'''

    def __init__(self, sample_size=128, num_seq=8, pred_step=3, network='resnet18'):
        super(CPC_Trans, self).__init__()
        torch.cuda.manual_seed(233)
        print('Using CPC with Transformer model')
        self.sample_size = sample_size
        self.num_seq = num_seq
        self.pred_step = pred_step
        self.last_size = int(math.ceil(sample_size / 32))   # 4

        self.backbone, self.param = select_resnet(network)
        self.param['num_layers'] = 1  # param for Transformer
        self.param['num_heads'] = 2  # param for Transformer
        self.param['num_cells'] = 1  # param for GRU
        self.param['hidden_size'] = self.param['feature_size']  # param for GRU

        self.transEnc = TransformerEncoders(d_model=self.param['feature_size'],
                                            N=self.param['num_layers'],
                                            h=self.param['num_heads'])
        self.agg = ConvGRU(input_size=self.param['feature_size'],
                           hidden_size=self.param['hidden_size'],
                           kernel_size=1,
                           num_layers=self.param['num_cells'])
        self.network_pred = nn.Sequential(
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(self.param['feature_size'], self.param['feature_size'], kernel_size=1, padding=0)
                            )
        self.mask = None
        self.relu = nn.ReLU(inplace=False)
        self._initialize_weights(self.transEnc)
        self._initialize_weights(self.agg)
        self._initialize_weights(self.network_pred)

    def forward(self, block):
        ### extract feature ###
        (B, N, C, H, W) = block.shape
        block = block.view(B * N, C, H, W)
        feature = self.backbone(block)
        del block

        feature_inf_all = feature.view(B, N, self.param['feature_size'], self.last_size, self.last_size)
        ### feed to transformer encoder, then predict future ###
        feature_to_trans = feature_inf_all.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, self.param['feature_size'])
        seq_trans = self.transEnc(feature_to_trans)
        seq_trans = seq_trans.view(B, N, self.last_size, self.last_size, self.param['feature_size'])
        seq_trans = seq_trans.permute(0, 1, 4, 2, 3).contiguous()
        feature_GT = seq_trans[:, N-self.pred_step::, :]
        seq_trans = self.relu(seq_trans)
        del feature_inf_all

        ### aggregate, predict future ###
        _, hidden = self.agg(seq_trans[:, 0:N-self.pred_step, :])
        hidden = hidden[:, -1, :]

        pred = []
        for i in range(self.pred_step):
            # sequentially pred future
            p_tmp = self.network_pred(hidden)
            pred.append(p_tmp)
            _, hidden = self.agg(self.relu(p_tmp).unsqueeze(1), hidden.unsqueeze(0))
            hidden = hidden[:, -1, :]
        pred = torch.stack(pred, 1)
        del hidden

        ### Get similarity score ###
        N = self.pred_step

        pred = pred.permute(0, 1, 3, 4, 2).contiguous().view(B * self.pred_step * self.last_size ** 2,
                                                             self.param['feature_size'])
        feature_GT = feature_GT.permute(0, 1, 3, 4, 2).contiguous().view(B * N * self.last_size ** 2,
                                                                           self.param['feature_size']).transpose(0, 1)
        score = torch.matmul(pred, feature_GT).view(B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2)
        del feature_GT, pred

        if self.mask is None:
            mask = torch.zeros((B, self.pred_step, self.last_size ** 2, B, N, self.last_size ** 2), dtype=torch.int8,
                               requires_grad=False).detach().cuda()

            mask[torch.arange(B), :, :, torch.arange(B), :, :] = -3  # spatial neg
            for k in range(B):
                mask[k, :, torch.arange(self.last_size ** 2), k, :,
                torch.arange(self.last_size ** 2)] = -1  # temporal neg
            tmp = mask.permute(0, 2, 1, 3, 5, 4).contiguous().view(B * self.last_size ** 2, self.pred_step,
                                                                   B * self.last_size ** 2, N)

            for j in range(B * self.last_size ** 2):
                tmp[j, torch.arange(self.pred_step), j, torch.arange(N - self.pred_step, N)] = 1  # pos
            mask = tmp.view(B, self.last_size ** 2, self.pred_step, B, self.last_size ** 2, N).permute(0, 2, 1, 3, 5, 4)
            self.mask = mask

        return [score, self.mask]

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.orthogonal_(param, 1)

    def reset_mask(self):
        self.mask = None


if __name__ == '__main__':
    model = CPC_Trans()
    model.to('cuda')

    input = torch.randn(4, 8, 3, 128, 128)
    input = input.to('cuda')
    output = model(input)
    print(output[0].shape, output[1].shape)
