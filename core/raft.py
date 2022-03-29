import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder, ResidualBlocksWithInputConv
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from utils.flow_warp import flow_warp
from residual_block import *
from upsample import PixelShufflePack

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.iters = 12
        # add number_blocks_reconstruction to args
        self.setting_2 = True
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = ResidualBlocksWithInputConv(3, 128, 5)
        self.cnet = ResidualBlocksWithInputConv(3, hdim+cdim, 5)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        # if args.small:
        #     self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
        #     self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
        #     self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
        # else:
        #     self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
        #     self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        #     self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)
        
        # reconstruction
        # in this case, each iteration has an individual reconstruction module
        if self.setting_2:
            self.reconstruct_list = nn.ModuleList()
            for i in range(self.iters):
                self.reconstruct_list.append(
                                        nn.Sequential(
                                            nn.Conv2d(2 * 3, hdim, 3, 1, 1),  
                                            make_layer(
                                                ResidualBlockNoBN,
                                                5,
                                                mid_channels=hdim),
                                            PixelShufflePack(
                                                hdim, hdim, 2, upsample_kernel=3),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            PixelShufflePack(
                                                hdim, 64, 2, upsample_kernel=3),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(64, 64, 3, 1, 1),
                                            nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                            nn.Conv2d(64, 3, 3, 1, 1),
                                        )
                )
            
        else: 
            self.conv_first = nn.Conv2d(2 * 3, hdim, 3, 1, 1)
            self.reconstruction = make_layer(
                ResidualBlockNoBN,
                5,
                mid_channels=hdim)
            # upsample
            self.upsample1 = PixelShufflePack(
                hdim, hdim, 2, upsample_kernel=3)
            self.upsample2 = PixelShufflePack(
                hdim, 64, 2, upsample_kernel=3)
            # we fix the output channels in the last few layers to 64.
            self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            # activation function
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=False)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        coords1 = coords_grid(N, H, W, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim

        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet(image1)
            fmap2 = self.fnet(image2)
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        output_predictions = []
        base = self.img_upsample(image1)
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            # if up_mask is None:
            #     flow_up = upflow8(coords1 - coords0)
            # else:
            #     flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow = coords1 - coords0
            image_warp2 = flow_warp(image2, flow.permute(0, 2, 3, 1))
            # reconstruction
            hr = torch.cat((image1, image_warp2), dim=1)
            if self.setting_2:
                hr = self.reconstruct_list[itr](hr)
            else:
                hr = self.conv_first(hr)
                hr = self.reconstruction(hr)
                hr = self.lrelu(self.upsample1(hr))
                hr = self.lrelu(self.upsample2(hr))
                hr = self.lrelu(self.conv_hr(hr))
                hr = self.conv_last(hr)
            hr += base
            output_predictions.append(hr)

        if test_mode:
            return output_predictions[-1]
            
        return output_predictions
