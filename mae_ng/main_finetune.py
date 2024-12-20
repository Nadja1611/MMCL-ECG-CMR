# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
from typing import Tuple
import numpy as np
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Subset, ConcatDataset
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import wandb

import timm

# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

import models_vit

from engine_finetune import train_one_epoch, evaluate

from util.dataset import SignalDataset


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    # Basic parameters
    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch200', type=str, metavar='MODEL',
                        help='Name of model to train')
    
    parser.add_argument('--input_channels', type=int, default=5, metavar='N',
                        help='input channels')
    parser.add_argument('--input_electrodes', type=int, default=65, metavar='N',
                        help='input electrodes')
    parser.add_argument('--time_steps', type=int, default=37000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(5, 65, 37000), type=Tuple,
                        help='images input size')

    parser.add_argument('--patch_height', type=int, default=65, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=200, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(65, 200), type=Tuple,
                        help='patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    # Augmentation parameters
    parser.add_argument('--masking_blockwise', action='store_true', default=False,
                        help='Masking blockwise in channel and time dimension (instead of random masking)')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_c_ratio', default=0.0, type=float,
                        help='Masking ratio in channel dimension (percentage of removed patches)')
    parser.add_argument('--mask_t_ratio', default=0.0, type=float,
                        help='Masking ratio in time dimension (percentage of removed patches)')

    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')
    parser.add_argument('--freq_shift_delta', default=0.005, type=float,
                        help='Delta for the frequency shift (default: 0.005)')

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 4')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Criterion parameters
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--attention_pool', action='store_true', default=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--downstream_task', default='classification', type=str,
                        help='downstream task (default: classification)')

    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path (default: None)')
    parser.add_argument('--labels_path', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--labels_mask_path', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path (default: None)')
    parser.add_argument('--val_labels_path', default='', type=str,
                        help='validation labels path (default: None)')
    parser.add_argument('--val_labels_mask_path', default='', type=str,
                        help='validation labels path (default: None)')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--pos_label', default=0, type=int,
                        help='classification type with the smallest count')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs',
                        help='path where to tensorboard log')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', default='',
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--plot_attention_map', action='store_true', default=False)
    parser.add_argument('--plot_embeddings', action='store_true', default=False)
    parser.add_argument('--embeddings_dir', default='',
                        help='path where to save embeddings, empty for no saving')
    parser.add_argument('--predictions_dir', default='',
                        help='path where to save prediction, empty for no saving')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def attention_forward_wrapper(attn_obj):
    """
    Modified version of def forward() of class Attention() in timm.models.vision_transformer
    """
    def my_forward(x):
        B, N, C = x.shape # C = embed_dim
        # (3, B, Heads, N, head_dim)
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        # (B, Heads, N, N)
        attn_obj.attn_map = attn # this was added 

        # (B, N, Heads*head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

def main(args):
    args.input_size = (args.input_channels, args.input_electrodes, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    if args.attention_pool:
        args.global_pool = "attention_pool"

    # misc.init_distributed_mode(args)
    args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_train = SignalDataset(data_path=args.data_path, labels_path=args.labels_path, labels_mask_path=args.labels_mask_path,
                                  downstream_task=args.downstream_task, augment=True, args=args)
    dataset_val = SignalDataset(data_path=args.val_data_path, labels_path=args.val_labels_path, labels_mask_path=args.val_labels_mask_path, 
                                downstream_task=args.downstream_task, transform=True, augment=False, args=args)

    # train balanced
    class_weights = 2.0 / (2.0 * torch.Tensor([1.0, 1.0])) # total_nb_samples / (nb_classes * samples_per_class)
    # train unbalanced
    # class_weights = 28030.0 / (2.0 * torch.Tensor([26015.0, 2015.0])) # CAD total_nb_samples / (nb_classes * samples_per_class)
    # class_weights = 5130.0 / (2.0 * torch.Tensor([2565.0, 2565.0])) # BM total_nb_samples / (nb_classes * samples_per_class) 

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None: # and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

        if args.wandb == True:
            config = vars(args)
            if args.wandb_id:
                wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity="madja161n")
            else:
                wandb.init(project=args.wandb_project, config=config, entity="madja161n")
    elif args.eval and "checkpoint" not in args.resume.split("/")[-1]:
        log_writer = SummaryWriter(log_dir=args.log_dir + "/eval")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        patch_size=args.patch_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        downstream_task=args.downstream_task,
        masking_blockwise=args.masking_blockwise,
        mask_ratio=args.mask_ratio,
        mask_c_ratio=args.mask_c_ratio,
        mask_t_ratio=args.mask_t_ratio
    )
    model.blocks[-1].attn.forward = attention_forward_wrapper(model.blocks[-1].attn) # required to read out the attention map of the last layer

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool == "attention_pool":
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias', 'attention_pool.out_proj.weight', 'attention_pool.in_proj_weight', 'attention_pool.out_proj.bias', 'attention_pool.in_proj_bias'}
            pass
        elif args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=0.01)#2e-5)

    # # for linear prob only
    # # hack: revise model's head with BN
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # # freeze all but the head
    # print("Every layer but the head frozen")
    # for _, p in model.named_parameters():
    #     p.requires_grad = False
    # for _, p in model.head.named_parameters():
    #     p.requires_grad = True
    
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 4

    print("base lr: %.2e" % (args.lr * 4 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    class_weights = class_weights.to(device=device)
    if args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()
    elif mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing) #LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("criterion = %s" % str(criterion))

    if not args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        sub_strings = args.resume.split("/")
        if "checkpoint" in sub_strings[-1]:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1])+1

        for epoch in range(0, nb_ckpts):
            if "checkpoint" not in sub_strings[-1]:
                args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"

            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
            if args.downstream_task == 'classification':
                print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val)} test images: {test_stats['acc']:.2f}% / {test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['auprc']:.2f}%")
            elif args.downstream_task == 'regression':
                print(f"Root Mean Squared Error (RMSE) / Pearson Correlation Coefficient (PCC) of the network on the {len(dataset_val)} test images:\
                     {test_stats['rmse']:.4f} / {test_stats['pcc']:.4f}")
        
            if args.wandb:
                wandb.log(test_history)

        exit(0)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)
    
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_stats = {'loss':np.inf, 'acc':0.0, 'f1':0.0, 'auroc':0.0, 'auprc':0.0, 'rmse':np.inf, 'pcc':0.0}
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch == 2:
        #     for _, p in model.named_parameters():
        #         p.requires_grad = True
        yo = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"# REQUIRED PARAMS: {len(yo)}")

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, train_history = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        test_stats, test_history = evaluate(data_loader_val, model, device, epoch, log_writer=log_writer, args=args)
        if args.downstream_task == 'classification':
            eval_criterion = "auroc"
        elif args.downstream_task == 'regression':
            eval_criterion = "rmse"

        if eval_criterion == "loss" or eval_criterion == "rmse":
            if early_stop.evaluate_decreasing_metric(val_metric=test_stats[eval_criterion]):
                break
            if args.output_dir and test_stats[eval_criterion] <= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion)
        else:
            if early_stop.evaluate_increasing_metric(val_metric=test_stats[eval_criterion]):
                break
            if args.output_dir and test_stats[eval_criterion] >= best_stats[eval_criterion]:
                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats, evaluation_criterion=eval_criterion)

        best_stats['loss'] = min(best_stats['loss'], test_stats['loss'])
        if args.downstream_task == 'classification':
            print(f"Accuracy / F1 / AUROC / AUPRC of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}% / {test_stats['f1']:.1f}% / {test_stats['auroc']:.1f}% / {test_stats['auprc']:.1f}%")
            best_stats['acc'] = max(best_stats['acc'], test_stats["acc"])
            best_stats['f1'] = max(best_stats['f1'], test_stats['f1'])
            best_stats['auroc'] = max(best_stats['auroc'], test_stats['auroc'])
            best_stats['auprc'] = max(best_stats['auprc'], test_stats['auprc'])
            print(f'Max Accuracy / F1 / AUROC / AUPRC: {best_stats["acc"]:.2f}% / {best_stats["f1"]:.2f}% / {best_stats["auroc"]:.2f}% / {best_stats["auprc"]:.2f}%\n')
        elif args.downstream_task == 'regression':
            print(f"Root Mean Squared Error (RMSE) / Pearson Correlation Coefficient (PCC) of the network on the {len(dataset_val)} test images:\
                 {test_stats['rmse']:.4f} / {test_stats['pcc']:.4f}")
            best_stats['rmse'] = min(best_stats['rmse'], test_stats['rmse'])
            best_stats['pcc'] = max(best_stats['pcc'], test_stats['pcc'])
            print(f'Min Root Mean Squared Error (RMSE) / Max Pearson Correlation Coefficient: {best_stats["rmse"]:.4f} / {best_stats["pcc"]:.4f}\n')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.wandb:
            wandb.log(train_history | test_history)

    if args.wandb:
        wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
