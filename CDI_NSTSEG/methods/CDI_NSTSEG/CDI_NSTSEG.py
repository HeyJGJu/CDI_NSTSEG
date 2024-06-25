import numpy as np

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample


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
            # ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=2048, out_dim=out_c),
        )
        self.c4_down = nn.Sequential(ConvBNReLU(1024, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(512, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(256, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))

    def forward(self, xs):
        #print(xs)
        #print(len(xs))
        #print(type(xs))
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        '''''
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        x1 = c1.shape
        print('x1')
        print(x1)
        x2 = c2.shape
        print('x2')
        print(x2)
        x3 = c3.shape
        print('x3')
        print(x3)
        x4 = c4.shape
        print('x4')
        print(x4)
        x5 = c5.shape
        print('x5')
        print(x5)
        '''''
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
        self.c5_down = nn.Sequential(ConvBNReLU(64, 1024, 3, 1, 1))
        self.c4_down = nn.Sequential(ConvBNReLU(64, 512, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(64, 256, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(64, 128, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, 64, 3, 1, 1))

    def forward(self, l, m, s, return_feats=False):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        #rectangled = []
        tgt_size = m.shape[2:]
        # 尺度缩小
        l = self.conv_l_pre_down(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        l = self.conv_l_post_down(l)
        # 尺度不变
        m = self.conv_m(m)
        # 尺度增加(这里使用上采样之后卷积的策略)
        s = self.conv_s_pre_up(s)
        s = cus_sample(s, mode="size", factors=m.shape[2:])
        s = self.conv_s_post_up(s)
        attn = self.trans(torch.cat([l, m, s], dim=1))
        attn_l, attn_m, attn_s = torch.softmax(attn, dim=1).chunk(3, dim=1)
        lms = attn_l * l + attn_m * m + attn_s * s
        #rectangled.append(lms)
        #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        #c1, c2, c3, c4, c5 = rectangled
        #x1 = c1.shape
        #print('x1')
        #print(x1)


        if return_feats:
            return lms, dict(attn_l=attn_l, attn_m=attn_m, attn_s=attn_s, l=l, m=m, s=s)
        return lms
        

        '''''
        c1, c2, c3, c4, c5 = lms
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        x1 = c1.shape
        print('x1')
        print(x1)
        x2 = c2.shape
        print('x2')
        print(x2)
        x3 = c3.shape
        print('x3')
        print(x3)
        x4 = c4.shape
        print('x4')
        print(x4)
        x5 = c5.shape
        print('x5')
        print(x5)
        e5 = self.c5_down(c5)
        e4 = self.c4_down(c4)
        e3 = self.c3_down(c3)
        e2 = self.c2_down(c2)
        e1 = self.c1_down(c1)
        g1 = e1.shape
        print('g1')
        print(g1)
        g2 = e2.shape
        print('g2')
        print(g2)
        g3 = e3.shape
        print('g3')
        print(g3)
        g4 = e4.shape
        print('g4')
        print(g4)
        g5 = e5.shape
        print('g5')
        print(g5)
        '''

        #return c5, c4, c3, c2, c1
        #return lms
        
'''''
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        #print('lms')
        #print(lms)
        x1 = lms.shape#################
        print('x1')
        print(x1)######
        return lms
'''''
    
'''''
class TransLayer_yuan(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        
        self.c5_down = nn.Sequential(
            #ConvBNReLU(2048, 256, 3, 1, 1),
            last_module(in_dim=64, out_dim=out_c),
        )
        
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        self.c4_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        self.c3_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        self.c2_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        self.c1_down = nn.Sequential(ConvBNReLU(64, out_c, 3, 1, 1))
        self.c = [64, 64, 64, 64, 64]

    def forward(self, xs):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        rectangled = []
        for i in self.c:
            in_c = self.c[i]
            lms = SIU(in_dim=in_c)
            rectangled.append(lms)
        c1, c2, c3, c4, c5 = rectangled
        x1 = c1.shape
        print('x1')
        print(x1)
        #return c5, c4, c3, c2, c1



        print(xs)
        assert isinstance(xs, (tuple, list))
        assert len(xs) == 5
        c1, c2, c3, c4, c5 = xs
        c5 = self.c5_down(c5)
        c4 = self.c4_down(c4)
        c3 = self.c3_down(c3)
        c2 = self.c2_down(c2)
        c1 = self.c1_down(c1)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        x1 = c1.shape
        print('x1')
        print(x1)
        x2 = c2.shape
        print('x2')
        print(x2)
        x3 = c3.shape
        print('x3')
        print(x3)
        x4 = c4.shape
        print('x4')
        print(x4)
        x5 = c5.shape
        print('x5')
        print(x5)
'''



class HMU(nn.Module):
    def __init__(self, in_c, num_groups=4, hidden_dim=None):
        super().__init__()
        self.num_groups = num_groups

        hidden_dim = hidden_dim or in_c // 2
        expand_dim = hidden_dim * num_groups
        self.expand_conv = ConvBNReLU(in_c, expand_dim, 1)
        self.gate_genator = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(num_groups * hidden_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, num_groups * hidden_dim, 1),
            nn.Softmax(dim=1),
        )

        self.interact = nn.ModuleDict()
        self.interact["0"] = ConvBNReLU(hidden_dim, 3 * hidden_dim, 3, 1, 1)
        for group_id in range(1, num_groups - 1):
            self.interact[str(group_id)] = ConvBNReLU(2 * hidden_dim, 3 * hidden_dim, 3, 1, 1)
        self.interact[str(num_groups - 1)] = ConvBNReLU(2 * hidden_dim, 2 * hidden_dim, 3, 1, 1)

        self.fuse = nn.Sequential(nn.Conv2d(num_groups * hidden_dim, in_c, 3, 1, 1), nn.BatchNorm2d(in_c))
        self.final_relu = nn.ReLU(True)

    def forward(self, x):
        xs = self.expand_conv(x).chunk(self.num_groups, dim=1)

        outs = []

        branch_out = self.interact["0"](xs[0])
        outs.append(branch_out.chunk(3, dim=1))

        for group_id in range(1, self.num_groups - 1):
            branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
            outs.append(branch_out.chunk(3, dim=1))

        group_id = self.num_groups - 1
        branch_out = self.interact[str(group_id)](torch.cat([xs[group_id], outs[group_id - 1][1]], dim=1))
        outs.append(branch_out.chunk(2, dim=1))

        out = torch.cat([o[0] for o in outs], dim=1)
        gate = self.gate_genator(torch.cat([o[-1] for o in outs], dim=1))
        out = self.fuse(out * gate)
        return self.final_relu(out + x)


def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()


@MODELS.register()
class ZoomNet(BasicModelClass):
    def __init__(self):
        super().__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.translayer = TransLayer(out_c=64)  # [c5, c4, c3, c2, c1]
        #self.translayer_yuan = TransLayer_yuan(out_c=128)
        #self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])
        self.merge_layers = nn.ModuleList([SIU(in_dim=in_c) for in_c in (64, 64, 64, 64, 64)])

        self.d5 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d4 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d3 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d2 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))
        self.d1 = nn.Sequential(HMU(64, num_groups=6, hidden_dim=32))

        self.e5_down = nn.Sequential(ConvBNReLU(64, 1024, 3, 1, 1))
        self.e4_down = nn.Sequential(ConvBNReLU(64, 512, 3, 1, 1))
        self.e3_down = nn.Sequential(ConvBNReLU(64, 256, 3, 1, 1))
        self.e2_down = nn.Sequential(ConvBNReLU(64, 128, 3, 1, 1))
        self.e1_down = nn.Sequential(ConvBNReLU(64, 64, 3, 1, 1))
        
        self.out_layer_00 = ConvBNReLU(64, 32, 3, 1, 1)
        self.out_layer_01 = nn.Conv2d(32, 1, 1)

    def encoder_translayer(self, x):
        en_feats = self.shared_encoder(x)
        trans_feats = self.translayer(en_feats)############
        return trans_feats

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = self.encoder_translayer(l_scale)
        m_trans_feats = self.encoder_translayer(m_scale)
        s_trans_feats = self.encoder_translayer(s_scale)

        feats = []
        for l, m, s, layer in zip(l_trans_feats, m_trans_feats, s_trans_feats, self.merge_layers):
            siu_outs = layer(l=l, m=m, s=s)
            feats.append(siu_outs)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######\
        x1 = self.e1_down(feats[0])
        c1 = x1.shape
        print('c1')
        print(c1)
        x2 = self.e2_down(feats[1])
        c2 = x2.shape
        print('c2')
        print(c2)
        x3 = self.e3_down(feats[2])
        c3 = x3.shape
        print('c3')
        print(c3)
        x4 = self.e4_down(feats[3])
        c4 = x4.shape
        print('c4')
        print(c4)
        x5 = self.e5_down(feats[4])
        c5 = x5.shape
        print('c5')
        print(c5)

        x = self.d5(feats[0])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + feats[1])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + feats[2])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + feats[3])
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + feats[4])
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return dict(seg=logits)

    def train_forward(self, data, **kwargs):
        assert not {"image1.5", "image1.0", "image0.5", "mask"}.difference(set(data)), set(data)

        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(
            l_scale=data["image1.5"],
            m_scale=data["image1.0"],
            s_scale=data["image0.5"],
        )
        return output["seg"]

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])

            sod_loss = F.binary_cross_entropy_with_logits(input=preds, target=resized_gts, reduction="mean")
            losses.append(sod_loss)
            loss_str.append(f"{name}_BCE: {sod_loss.item():.5f}")

            ual_loss = cal_ual(seg_logits=preds, seg_gts=resized_gts)
            ual_loss *= ual_coef
            losses.append(ual_loss)
            loss_str.append(f"{name}_UAL_{ual_coef:.5f}: {ual_loss.item():.5f}")
        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups


@MODELS.register()
class ZoomNet_CK(ZoomNet):
    def __init__(self):
        super().__init__()
        self.dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def encoder(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x0, x1, x2, x3, x4 = self.shared_encoder(x)
        return x0, x1, x2, x3, x4

    def trans(self, x0, x1, x2, x3, x4):
        x5, x4, x3, x2, x1 = self.translayer([x0, x1, x2, x3, x4])#####################
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        x1 = x0.shape
        print('x1')
        print(x1)
        x2 = x1.shape
        print('x2')
        print(x2)
        x3 = x2.shape
        print('x3')
        print(x3)
        x4 = x3.shape
        print('x4')
        print(x4)
        x5 = x4.shape
        print('x5')
        print(x5)
        return x5, x4, x3, x2, x1
    
    def deco(self, x5, x4, x3, x2, x1):
        c5, c4, c3, c2, c1 = self.translayer_yuan([x5, x4, x3, x2, x1])
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')######
        e1 = c1.shape
        print('e1')
        print(e1)
        e2 = c2.shape
        print('e2')
        print(e2)
        e3 = c3.shape
        print('e3')
        print(e3)
        e4 = c4.shape
        print('e4')
        print(e4)
        e5 = e4.shape
        print('e5')
        print(e5)
        return c5, c4, c3, c2, c1


    def decoder(self, x5, x4, x3, x2, x1):
        x = self.d5(x5)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d4(x + x4)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d3(x + x3)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d2(x + x2)
        x = cus_sample(x, mode="scale", factors=2)
        x = self.d1(x + x1)
        x = cus_sample(x, mode="scale", factors=2)
        logits = self.out_layer_01(self.out_layer_00(x))
        return logits

    def body(self, l_scale, m_scale, s_scale):
        l_trans_feats = checkpoint(self.encoder, l_scale, self.dummy)
        m_trans_feats = checkpoint(self.encoder, m_scale, self.dummy)
        s_trans_feats = checkpoint(self.encoder, s_scale, self.dummy)
        l_trans_feats = checkpoint(self.trans, *l_trans_feats)
        m_trans_feats = checkpoint(self.trans, *m_trans_feats)
        s_trans_feats = checkpoint(self.trans, *s_trans_feats)

        feats = []
        for layer_idx, (l, m, s) in enumerate(zip(l_trans_feats, m_trans_feats, s_trans_feats)):
            siu_outs = checkpoint(self.merge_layers[layer_idx], l, m, s)
            feats.append(siu_outs)

        logits = checkpoint(self.decoder, *feats)
        return dict(seg=logits)
