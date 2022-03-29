import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate
from builder import build_dataloader, build_dataset
from metrics import psnr, ssim

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_REDS(model, cfg):
    model.eval()
    val_dataset = build_dataset(cfg.data.test)
    PSNR_sum = 0.0
    SSIM_sum = 0.0
    for val_id in range(len(val_dataset)):
        data_blob = val_dataset[val_id]
        image1 = data_blob['lq'][0]
        image2 = data_blob['lq'][1]
        gt = data_blob['gt'][0]
        image1 = image1[None, ...].cuda()
        image2 = image2[None, ...].cuda()
        gt = gt[None, ...].cuda()
        input_data = torch.stack([image1, image2], dim=1)
        output = test_big_size(model, input_data)
        gt = gt.detach().cpu().numpy()
        gt = gt[0] *255
        output = output[0] * 255
        PSNR_sum += psnr(gt, output, crop_border=0, input_order='CHW')
        SSIM_sum += ssim(gt, output, crop_border=0, input_order='CHW')
    return PSNR_sum/len(val_dataset), SSIM_sum/len(val_dataset)

@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

def test_big_size(model, input_data, patch_h=64, patch_w=64,
                        patch_h_overlap=32, patch_w_overlap=32):
    # input_data shape n, t, c, h, w 
    # output shape n, c, h, w
    scale = 4
    H = input_data.shape[3]
    W = input_data.shape[4]
    t = input_data.shape[1]
    center_idx = int(t/2)
    test_result = np.zeros((input_data.shape[0], 3, scale*H, scale*W))
    h_index = 1
    while (patch_h*h_index-patch_h_overlap*(h_index-1)) < H:
        test_horizontal_result = np.zeros((input_data.shape[0],
                                            3, scale*patch_h, scale*W))
        h_begin = patch_h*(h_index-1)-patch_h_overlap*(h_index-1)
        h_end = patch_h*h_index-patch_h_overlap*(h_index-1)
        w_index = 1
        w_end = 0
        while (patch_w*w_index-patch_w_overlap*(w_index-1)) < W:
            w_begin = patch_w*(w_index-1)-patch_w_overlap*(w_index-1)
            w_end = patch_w*w_index-patch_w_overlap*(w_index-1)
            test_patch = input_data[:, :, :, h_begin:h_end, w_begin:w_end]
            output_patch = model(test_patch[:, 0, :, :, :], test_patch[:, 1, :, :, :])[-1]
            output_patch = \
                output_patch.cpu().detach().numpy().astype(np.float32)
            if w_index == 1:
                test_horizontal_result[:, :, :, w_begin*scale:w_end*scale] = \
                    output_patch
            else:
                for i in range(patch_w_overlap*scale):
                    test_horizontal_result[:, :, :, w_begin * scale + i] = \
                        test_horizontal_result[:, :, :, w_begin * scale + i]\
                        * (patch_w_overlap * scale-1-i)/(patch_w_overlap * scale -1)\
                        + output_patch[:, :, :, i] * i/(patch_w_overlap * scale -1)
                cur_begin = w_begin+patch_w_overlap
                cur_begin = cur_begin*scale
                test_horizontal_result[:, :, :, cur_begin:w_end*scale] = \
                    output_patch[:, :, :, patch_w_overlap * scale:]
            w_index += 1
        test_patch = input_data[:, :, :, h_begin:h_end, -patch_w:]
        output_patch = model(test_patch[:, 0, :, :, :], test_patch[:, 1, :, :, :])[-1]
        output_patch = \
            output_patch.cpu().detach().numpy().astype(np.float32)
        output_patch = output_patch[:, :, :, :]
        last_range = w_end-(W-patch_w)
        last_range = last_range * scale

        for i in range(last_range):
            term1 = test_horizontal_result[:, :, :, W*scale-patch_w*scale+i]
            rate1 = (last_range-1-i)/(last_range-1)
            term2 = output_patch[:, :, :, i]
            rate2 = i/(last_range-1)
            test_horizontal_result[:, :, :, W*scale-patch_w*scale+i] = \
                term1*rate1+term2*rate2
        test_horizontal_result[:, :, :, w_end*scale:] = \
            output_patch[:, :, :, last_range:]

        if h_index == 1:
            test_result[:, :, h_begin*scale:h_end*scale, :] = test_horizontal_result
        else:
            for i in range(patch_h_overlap*scale):
                term1 = test_result[:, :, h_begin*scale+i, :]
                rate1 = (patch_h_overlap*scale-1-i)/(patch_h_overlap*scale-1)
                term2 = test_horizontal_result[:, :, i, :]
                rate2 = i/(patch_h_overlap*scale-1)
                test_result[:, :, h_begin*scale+i, :] = \
                    term1 * rate1 + term2 * rate2
            test_result[:, :, h_begin*scale+patch_h_overlap*scale:h_end*scale, :] = \
                test_horizontal_result[:, :, patch_h_overlap*scale:, :]
        h_index += 1

    test_horizontal_result = np.zeros((input_data.shape[0], 3, patch_h*scale, W*scale))
    w_index = 1
    while (patch_w * w_index - patch_w_overlap * (w_index-1)) < W:
        w_begin = patch_w * (w_index-1) - patch_w_overlap * (w_index-1)
        w_end = patch_w * w_index - patch_w_overlap * (w_index-1)
        test_patch = input_data[:, :, :, -patch_h:, w_begin:w_end]
        output_patch = model(test_patch[:, 0, :, :, :], test_patch[:, 1, :, :, :])[-1]
        output_patch = \
            output_patch.cpu().detach().numpy().astype(np.float32)
        output_patch = output_patch
        if w_index == 1:
            test_horizontal_result[:, :, :, w_begin*scale:w_end*scale] = output_patch
        else:
            for i in range(patch_w_overlap*scale):
                term1 = test_horizontal_result[:, :, :, w_begin*scale+i]
                rate1 = (patch_w_overlap*scale-1-i)/(patch_w_overlap*scale-1)
                term2 = output_patch[:, :, :, i]
                rate2 = i/(patch_w_overlap*scale-1)
                test_horizontal_result[:, :, :, w_begin*scale+i] = \
                    term1*rate1+term2*rate2
            cur_begin = w_begin+patch_w_overlap
            test_horizontal_result[:, :, :, cur_begin*scale:w_end*scale] = \
                output_patch[:, :, :, patch_w_overlap*scale:]
        w_index += 1
    test_patch = input_data[:, :, :,  -patch_h:, -patch_w:]
    output_patch = model(test_patch[:, 0, :, :, :], test_patch[:, 1, :, :, :])[-1]
    output_patch = output_patch.cpu().detach().numpy().astype(np.float32)
    output_patch = output_patch
    last_range = w_end-(W-patch_w)
    for i in range(last_range*scale):
        term1 = test_horizontal_result[:, :, :, W*scale-patch_w*scale+i]
        rate1 = (last_range*scale-1-i)/(last_range*scale-1)
        term2 = output_patch[:, :, :, i]
        rate2 = i/(last_range*scale-1)
        test_horizontal_result[:, :, :, W*scale-patch_w*scale+i] = \
            term1*rate1+term2*rate2
    test_horizontal_result[:, :, :, w_end*scale:] = \
        output_patch[:, :, :, last_range*scale:]

    last_last_range = h_end-(H-patch_h)
    for i in range(last_last_range*scale):
        term1 = test_result[:, :, H*scale-patch_w*scale+i, :]
        rate1 = (last_last_range*scale-1-i)/(last_last_range*scale-1)
        term2 = test_horizontal_result[:, :, i, :]
        rate2 = i/(last_last_range*scale-1)
        test_result[:, :, H*scale-patch_w*scale+i, :] = \
            term1*rate1+term2*rate2
    cur_result = test_horizontal_result[:, :, last_last_range*scale:, :]
    test_result[:, :, h_end*scale:, :] = cur_result
    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)


