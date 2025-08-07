import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    gt_dir = osp.join(root_dir, 'gt')
    pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')
    
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


def evaluate():
    class Config:
        methods = None
        datasets = None
        root_dir = "./"
        save_dir = None
        cuda = True

    config = Config()
    main(config)