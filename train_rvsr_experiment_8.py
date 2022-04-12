from __future__ import print_function, division
import sys

from core.utils.flow_viz import flow_uv_to_colors
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

from torch.utils.data import DataLoader
from rvsr import RVSR
from utils.flow_viz import flow_to_image
from evaluate import *
import datasets

from builder import build_dataloader, build_dataset
from mmcv import Config
import sr_reds_multiple_gt_dataset

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

os.environ["CUDA_VISIBLE_DEVICES"]="3"
# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 2000
LOG_FREQ = 500
LOG_PATH = "/home1/xinyuan/work/RAFT_result"
cfg = Config.fromfile("/home1/xinyuan/work/RAFT/config/raft_multi_reds4_update.py")

def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps)


def sequence_loss(pred_imgs, gt_img, gamma=0.8):
    # (n, c, h, w)
    n_predictions = len(pred_imgs)    
    total_loss = 0.0

    # TODO: should we exlude invalid pixels and extremely large diplacements?
    
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        for j in range(15):
            if pred_imgs[i][j] is not None:
                i_loss = charbonnier_loss(pred_imgs[i][j], gt_img[:, j, :, :, :])
                total_loss += i_weight * (i_loss).mean()

    return total_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, log_dir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.log_dir = log_dir

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter(self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RVSR(args,  "/home1/xinyuan/work/RAFT/models/raft-things.pth"), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #     model.module.freeze_bn()
    
    dataset = build_dataset(cfg.data.train)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    task_name = "experiment_8"
    full_log_path = os.path.join(LOG_PATH, task_name)
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, full_log_path)

    add_noise = True
    input_num = args.input_frame
    input_num = 15
    fix_iter = 50000
    # refer_idx = (input_num+1)/2
    # refer_idx = 0
    # refer_idx = int(refer_idx)
    should_keep_training = True
    loss_sum = 0
    while should_keep_training:

        for i_batch, data_blob in enumerate(dataset):
            optimizer.zero_grad()
            input_frames = data_blob['lq'][:input_num]
            gt = data_blob['gt'][:input_num]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                for i in range(input_num):
                    input_frames[i] = (input_frames[i]
                                     + stdv * torch.randn(*input_frames[i].shape).cuda()).clamp(0.0, 255.0)

            input_frames = input_frames[None, ...]
            gt = gt[None, ...]
            gt = gt.cuda()

            if total_steps < fix_iter:
                for k, v in model.named_parameters():
                    if 'RAFT' in k:
                        v.requires_grad_(False)
            else:
                model.requires_grad_(True)
            output_predictions = model(input_frames, iters=args.iters)
            # n, t, c, h, w
            loss= sequence_loss(output_predictions, gt, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push({'loss':loss})
            # logger.push(metrics)
            if total_steps % LOG_FREQ == LOG_FREQ - 1:
                output = []
                _, c, h, w = output_predictions[0][1].size()
                for i in range(25):
                    for j in range(15):
                        if output_predictions[i][j] is None:
                            output.append(output_predictions[0][1].new_zeros(c, h, w))
                        else:
                            output.append(torch.clamp(output_predictions[i][j][0], 0.0, 1.0))
                grid = torchvision.utils.make_grid(output, nrow=15)
                logger.writer.add_image('prediction_list', grid, total_steps)
                image_list = []
                for i in range(input_num):
                    image_list.append(input_frames[0, i, :, :, :])
                grid = torchvision.utils.make_grid(image_list)
                logger.writer.add_image('input', grid, total_steps)
                error_first = abs(output_predictions[0][7][0]-gt[0, 7, :, :, :])
                error_last = abs(output_predictions[-1][7][0]-gt[0, 7, :, :, :])
                max_val = torch.max(error_first)
                error_first = error_first/max_val
                error_last = error_last/max_val
                grid = torchvision.utils.make_grid([gt[0, 7, :, :, :], error_first, error_last])
                logger.writer.add_image('error', grid, total_steps)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)
                output_list, psnr, ssim = validate_multi_REDS(model, input_num,  cfg, -1)
                grid = torchvision.utils.make_grid(output_list)
                logger.writer.add_image('validate_last', grid, total_steps)
                results = {'PSNR_last': psnr, 'SSIM_last': ssim}
                logger.write_dict(results)
                model.train()
            
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--input_frame', type=int, default=2)
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
