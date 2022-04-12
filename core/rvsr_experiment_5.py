import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.raft import RAFT
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
        self.reconstruct_list = nn.ModuleList()
        # add this in the config
        mid_channels = 64
        num_blocks = 30
        self.feature_extractor = ResidualBlocksWithInputConv(3, mid_channels, num_blocks)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)
        self.img_upsample_input = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=False)
        hdim = 128
        for i in range(self.iters):
            self.reconstruct_list.append(
                                    nn.Sequential(
                                        nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                        nn.Conv2d(64, 3, 3, 1, 1),
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
        for itr in range(iters):
            self.optical_flow_list.append(flow_output_list[itr])
            image_warp2 = flow_warp(image2, flow_output_list[itr].permute(0, 2, 3, 1))
            self.warped_img_list.append((image_warp2[0].detach().cpu() + 1.0)/2)
            fmap_warp2 = flow_warp(fmap2, flow_output_list[itr].permute(0, 2, 3, 1))
            # reconstruction
            hr = torch.cat([fmap1, fmap_warp2], dim=1)
            hr = self.reconstruct_list[itr](hr)
            hr += base
            output_predictions.append(hr)
        return output_predictions