
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import to_2tuple
#from methods.module.base_model import BasicModelClass
#from methods.module.conv_block import ConvBNReLU
#import methods.conv_block as ConvBNReLU
#from utils.builder import MODELS
from utils.ops import cus_sample
#import utils.ops.tensor_ops as cus_sample

#import backbone.resnet.resnet as resnet
import backbone.resnet as resnet
import cv2
import numpy as np
import matplotlib.pyplot as plt

def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        
        super().__init__()
        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        self.conv1 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=5, padding=5)
        self.conv4 = ConvBNReLU(in_dim, out_dim, kernel_size=3, dilation=7, padding=7)
        self.conv5 = ConvBNReLU(in_dim, out_dim, kernel_size=1)
        self.fuse = ConvBNReLU(5 * out_dim, out_dim, 3, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(cus_sample(x.mean((2, 3), keepdim=True), mode="size", factors=x.size()[2:]))
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class TransLayer(nn.Module):
    def __init__(self, out_c, last_module=ASPP):
       
        super().__init__()
        
        self.c5_down = nn.Sequential(
            
            last_module(in_dim=2048, out_dim=out_c),
        )
        
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        return c5, c4, c3, c2, c1





class SIU(nn.Module):
    def __init__(self, in_dim):
        
        super().__init__()
        
        self.conv_l_pre_down = ConvBNReLU(in_dim, in_dim, 5, stride=1, padding=2)
        self.conv_l_post_down = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_m = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.conv_s_post_up = ConvBNReLU(in_dim, in_dim, 3, 1, 1)
        self.trans = nn.Sequential(
            ConvBNReLU(3 * in_dim, in_dim, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            ConvBNReLU(in_dim, in_dim, 3, 1, 1),
            nn.Conv2d(in_dim, 3, 1),
        )

    def forward(self, l, m, s, return_feats=False):
        tgt_size = m.shape[2:]
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        m = self.conv_m(m)
        s = self.conv_s_pre_up(s)
        
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s

        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms

class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        
        return out

class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)

        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)

        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)

        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

class Positioning(nn.Module):
    def __init__(self, channel):
        super(Positioning, self).__init__()
        self.channel = channel
        self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)

    def forward(self, x):
        cab = self.cab(x)
        sab = self.sab(cab)
        map = self.map(sab)
        

        return sab, map

class Focus(nn.Module):
    def __init__(self, channel1, channel2):
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        self.fp = Context_Exploration_Block(self.channel1)
        self.fn = Context_Exploration_Block(self.channel1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()
    
    def forward(self, x, y, in_map):
        up = self.up(y)
        input_map = self.input_map(in_map)
        f_feature = x * input_map
        b_feature = x * (1 - input_map)
        fp = self.fp(f_feature)
        fn = self.fn(b_feature)
        refine1 = up - (self.alpha * fp)
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)
        refine2 = refine1 + (self.beta * fn)
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        output_map = self.output_map(refine2)
        return refine2, output_map
    
class CDI_NSTSEG(nn.Module):
    def __init__(self, backbone_path=None):
        super(CDI_NSTSEG, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])
        
        self.e5_down = nn.Sequential(ConvBNReLU(64, 512, 3, 1, 1))
        self.e4_down = nn.Sequential(ConvBNReLU(64, 256, 3, 1, 1))
        self.e3_down = nn.Sequential(ConvBNReLU(64, 128, 3, 1, 1))
        self.e2_down = nn.Sequential(ConvBNReLU(64, 64, 3, 1, 1))
        self.e1_down = nn.Sequential(ConvBNReLU(64, 32, 3, 1, 1))
        self.index = 0

        self.positioning = Positioning(512)
        
        self.focus3 = Focus(256, 512)
        
        self.focus2 = Focus(128, 256)
        
        self.focus1 = Focus(64, 128)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        
    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)
        
        return trans_feats

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = self.encoder_translayer(l_scale)
        
        m_trans_feats = self.encoder_translayer(m_scale)
        
        s_trans_feats = self.encoder_translayer(s_scale)
        

        feats = []
        for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
        
            siu_outs = layer(l=l, m=m, s=s)
            feats.append(siu_outs)
        
        
        x1 = self.e1_down(feats[4])
    
        x2 = self.e2_down(feats[3])
        
        x3 = self.e3_down(feats[2])
        
        x4 = self.e4_down(feats[1])
        
        x5 = self.e5_down(feats[0])
        
        return x1, x2, x3, x4, x5

    def forward(self, data, data_1, data_0, **kwargs):
        
        cr1, cr2, cr3, cr4, cr5 = self.body(
            l_scale=data_1,
            m_scale=data,
            s_scale=data_0,
            
        )
        
        
        positioning, predict4 = self.positioning(cr5)
    
        focus3, predict3 = self.focus3(cr4, positioning, predict4)
        focus2, predict2 = self.focus2(cr3, focus3, predict3)
        focus1, predict1 = self.focus1(cr2, focus2, predict2)
        
        predict4 = F.interpolate(predict4, size=data.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=data.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=data.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=data.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)
        
