import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from backbone.raft import RAFT
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from core.extractor import ResidualBlocksWithInputConv
from residual_block import *
from upsample import PixelShufflePack
from utils.flow_warp import flow_warp

class ModulatedDCNPack(ModulatedDeformConv2d):
    """Modulated Deformable Convolutional Pack.

    Different from the official DCN, which generates offsets and masks from
    the preceding features, this ModulatedDCNPack takes another different
    feature to generate masks and offsets.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True)
        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset, val=0, bias=0)

    def forward(self, x, extra_feat):
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

class RVSR(nn.Module):
    def __init__(self, args, raft_pretrained_path):
        super(RVSR, self).__init__()
        self.RAFT = RAFT(args)
        # self.RAFT.load_state_dict(torch.load(raft_pretrained_path), strict=False)
        checkpoint = torch.load(raft_pretrained_path)
        for key in list(checkpoint.keys()):
            if 'module.' in key:
                checkpoint[key.replace('module.', '')] = checkpoint[key]
                del checkpoint[key]

        self.RAFT.load_state_dict(checkpoint)
        self.iters = 12
        self.optical_flow_list = []
        self.warped_img_list = []
        self.fusion_list = nn.ModuleList()
        self.reconstruct_list = nn.ModuleList()
        # add this in the config
        self.mid_channels = 64
        num_blocks = 30
        self.feature_extractor = ResidualBlocksWithInputConv(3, self.mid_channels, num_blocks)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_input = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        hdim = 128
        for i in range(self.iters):
            self.fusion_list.append(
                                nn.Sequential(
                                    nn.Conv2d(128, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(64, 64, 3, 1, 1),
                                )
            )
            self.reconstruct_list.append(
                                    nn.Sequential(
                                        PixelShufflePack(self.mid_channels, self.mid_channels, 2, upsample_kernel=3),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        PixelShufflePack(self.mid_channels, self.mid_channels, 2, upsample_kernel=3),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(64, 3, 3, 1, 1),
                                    )
            )
        self.fusion = nn.Conv2d(
            self.mid_channels * 2, self.mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            self.mid_channels, self.mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            self.mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def init_weights(self):
        pass

    def compute_flow(self, input_frames):
        # the input is n, t, c, h, w
        # the output is flow_forward, flow_backward
        # both with the shape of n, t
        n, t, c, h, w = input_frames.size() 
        image_1 = input_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        image_2 = input_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        flow_backward_list = self.RAFT(image_1, image_2)
        flow_forward_list = self.RAFT(image_2, image_1)

        return flow_forward_list, flow_backward_list


    def forward(self, input_frames, iters=12, flow_init=None, upsample=True, test_mode=False):
        input_frames = (input_frames*2) - 1.0
        n, t, c, h, w = input_frames.shape

        # the output is a two dimision list, with the dimesion, (2*self.iters + 1) x t
        output_predictions = []
        # 1. feature directly from encoder
        # 2. feature after forward propagation
        # 3. feature after backward propagation
        feature_grid = [[], [], []]
        for i in range(t):
            fmap = self.feature_extractor(input_frames[:, i, :, :, :])
            feature_grid[0].append(fmap)
        
        # input_frames_up is used as input of optical flow
        input_frames_up = input_frames.reshape(-1, c, h, w)
        input_frames_up = self.img_upsample_input(input_frames_up)
        input_frames_up = input_frames_up.reshape(n, t, c, 8*h, 8*w)
        # flow_list is a lenth iter_num list
        # each element is a n*(t-1), 2, h, w size flow
        flow_forward_list, flow_backward_list = self.compute_flow(input_frames_up)
        for i in range(self.iters*2+1):
            output_predictions.append([])
        # base
        input_frames = (input_frames+1.0)/2
        input_frames = input_frames.reshape(-1, c, h, w)
        base = self.img_upsample(input_frames)
        input_frames = input_frames.reshape(n, t, c, h, w)
        base = base.reshape(n, t, c, 4*h, 4*w)
        for itr in range(iters):
            flow_forward_list[itr] = flow_forward_list[itr].reshape(n, t-1, 2, h, w)
            flow_backward_list[itr] = flow_backward_list[itr].reshape(n, t-1, 2, h, w)
        # achieve feature map for image1 and image2
        # backward-time propagation
        feat_prop = input_frames.new_zeros(n, self.mid_channels, h, w)
        for i in range(t-1, -1, -1):
            if i==(t-1):
                feature_grid[1].append(feature_grid[0][t-1])
                feat_prop = feature_grid[0][t-1]
                for itr in range(iters):
                    output_predictions[itr].append(None)
            else:
                merged_fmap = feature_grid[0][i]
                for itr in range(iters): 
                    fmap_warp2 = flow_warp(feat_prop, flow_backward_list[itr][:, i, :, :, :].permute(0, 2, 3, 1))
                    # reconstruction
                    hr = torch.cat([merged_fmap, fmap_warp2], dim=1)
                    hr = self.fusion_list[itr](hr)
                    feat_prop = hr
                    feature_grid[1].append(feat_prop)
                    output = hr + merged_fmap
                    output = self.reconstruct_list[itr](output)
                    output = output + base[:, i, :, :, :]
                    output_predictions[itr].append(output)
        
        for itr in range(iters):
            output_predictions[itr] = output_predictions[itr][::-1]
        feature_grid[1] = feature_grid[1][::-1]
        
        feat_prop = input_frames.new_zeros(n, self.mid_channels, h, w)
        for i in range(0, t):
            if i==0:
                feature_grid[2].append(feature_grid[0][0])
                feat_prop = feature_grid[0][0]
                for itr in range(iters):
                    output_predictions[iters + itr].append(None)
            else:
                merged_fmap = feature_grid[0][i]
                for itr in range(iters): 
                    fmap_warp2 = flow_warp(feat_prop, flow_forward_list[itr][:, i-1, :, :, :].permute(0, 2, 3, 1))
                    # reconstruction
                    hr = torch.cat([merged_fmap, fmap_warp2], dim=1)
                    hr = self.fusion_list[itr](hr)
                    feat_prop = hr + merged_fmap
                    feature_grid[2].append(feat_prop)
                    output = self.reconstruct_list[itr](feat_prop)
                    output = output + base[:, i, :, :, :]
                    output_predictions[iters + itr].append(output)
                
            output = torch.cat([feature_grid[1][i], feature_grid[2][i]], dim=1)
            output = self.lrelu(self.fusion(output))
            output = self.lrelu(self.upsample1(output))
            output = self.lrelu(self.upsample2(output))
            output = self.lrelu(self.conv_hr(output))
            output = self.conv_last(output)
            output += base[:, i, :, :, :]
            output_predictions[2*iters].append(output) 

        return output_predictions