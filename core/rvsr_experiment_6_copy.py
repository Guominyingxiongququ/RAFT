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
        self.deform_list = nn.ModuleList()
        # add this in the config
        mid_channels = 64
        num_blocks = 30
        max_residue_magnitude = 10
        self.feature_extractor = ResidualBlocksWithInputConv(3, mid_channels, num_blocks)
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
                                    PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(64, 64, 3, 1, 1),
                                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                    nn.Conv2d(64, 3, 3, 1, 1),
                                )
            )
            self.deform_list.append(
                                DeformableAlignment(
                                    2 * mid_channels,
                                    mid_channels,
                                    3,
                                    padding=1,
                                    deform_groups=16,
                                    max_residue_magnitude=max_residue_magnitude
                                )     
            )

    def init_weights(self):
        pass

    def forward(self, input_frames, iters=12, flow_init=None, upsample=True, test_mode=False):
        input_frames = (input_frames*2) - 1.0
        image1 = input_frames[:, 0, :, :, :]
        image2 = input_frames[:, 1, :, :, :]
        image1_up = self.img_upsample_input(image1)
        image2_up = self.img_upsample_input(image2)
        flow_output_list = self.RAFT(image1_up, image2_up, iters=iters, flow_init=flow_init, upsample=upsample, test_mode=test_mode)
        output_predictions = []
        base = self.img_upsample(image1)
        base = (base+1.0)/2
        # achieve feature map for image1 and image2
        fmap1 = self.feature_extractor(image1)
        fmap2 = self.feature_extractor(image2)
        self.optical_flow_list = []
        self.warped_img_list = []
        merged_fmap = fmap1
        for itr in range(iters):
            self.optical_flow_list.append(flow_output_list[itr])
            image_warp2 = flow_warp(image2, flow_output_list[itr].permute(0, 2, 3, 1))
            self.warped_img_list.append((image_warp2[0].detach().cpu() + 1.0)/2)
            fmap_warp2 = flow_warp(fmap2, flow_output_list[itr].permute(0, 2, 3, 1))
            # reconstruction
            hr = torch.cat([merged_fmap, fmap_warp2], dim=1)
            hr = self.deform_list[itr](fmap2, hr, flow_output_list[itr])
            hr = torch.cat([merged_fmap, hr], dim=1)
            hr = self.fusion_list[itr](hr)
            merged_fmap = hr + merged_fmap
            hr = self.reconstruct_list[itr](merged_fmap)
            hr += base
            output_predictions.append(hr)
        return output_predictions

class DeformableAlignment(ModulatedDeformConv2d):

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(DeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow.flip(1).repeat(1,
                                                    offset_1.size(1) // 2, 1,
                                                    1)
        offset_2 = offset_2 + flow.flip(1).repeat(1,
                                                    offset_2.size(1) // 2, 1,
                                                    1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)
        import pdb
        pdb.set_trace()
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias,
                                       self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)