#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
from functools import partial
import time
import torch
import torch.nn as nn
import wandb
from fastai.callbacks import SaveModelCallback
from wandb.fastai import WandbCallback
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim

from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss, GHMCLoss, FocalLoss


# with open('wandb_key.txt', 'r') as f:
#     os.environ["WANDB_API_KEY"] = f.read().strip()  # 官网给你的key
#     os.environ["WANDB_MODE"] = "offline"  # 离线


def main(args):
    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir
    cur_time = time.time()
    epochs = 25
    lr = 1e-1
    batch_size = 5
    num_workers = 4
    use_bce = True
    bce_lambda = 0.5
    use_dice = True
    dice_lambda = 1
    use_GHMC = False
    ghmc_lambda = 1
    ghmc_mmt = 0
    ghmc_bins = 10
    use_focal = False
    focal_lambda = 1
    focal_alpha = 1
    focal_gamma = 2
    use_prelu = True
    se_at_end = True
    se_throughout = True

    thresh = 0.1
    hyper_param_map = {
        'batch_size': batch_size,
        'use_bce': use_bce,
        'bce_lambda': bce_lambda,
        'use_dice': use_dice,
        'dice_lambda': dice_lambda,
        'use_GHMC': use_GHMC,
        'ghmc_lambda': ghmc_lambda,
        'ghmc_mmt': ghmc_mmt,
        'ghmc_bins': ghmc_bins,
        'use_focal': use_focal,
        'focal_lambda': focal_lambda,
        'focal_alpha': focal_alpha,
        'focal_gamma': focal_gamma,
        'optimizer': 'adam',
        'epochs': epochs,
        'lr': lr,
        'thresh': thresh,
        'use_prelu': use_prelu,
        'se_at_end': se_at_end,
        'se_throughout': se_throughout
    }
    history_path = os.path.join('/data/PyTorch_model/FracNet/history', str(cur_time))
    os.makedirs(history_path, exist_ok=True)
    with open(os.path.join(history_path, 'param.txt'), 'w') as f:
        f.write(str(hyper_param_map))
    wandb_dir = os.path.join('/data', 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project="ribfrac",
               name='FracNet_' + str(cur_time),
               dir=wandb_dir)
    wandb.config.update(hyper_param_map)

    losses = []
    if use_bce:
        losses += [nn.BCEWithLogitsLoss(), bce_lambda]
    if use_dice:
        losses += [DiceLoss(), dice_lambda]
    if use_GHMC:
        losses += [GHMCLoss(ghmc_mmt, ghmc_bins), ghmc_lambda]
    if use_focal:
        losses += [FocalLoss(focal_alpha, focal_gamma), focal_lambda]
    optimizer = optim.Adam
    criterion = MixLoss(*losses)

    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, first_out_channels=16, use_prelu=use_prelu, se_at_end=se_at_end, se_throughout=se_throughout)
    model = nn.DataParallel(model.cuda())

    transforms = [  # -100 1000
        tsfm.Window(-100, 1000),
        tsfm.MinMaxNorm(-100, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
                                   transforms=transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, True,
                                                  num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
                                 transforms=transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
                                                num_workers)

    databunch = DataBunch(dl_train, dl_val,
                          collate_fn=FracNetTrainDataset.collate_fn)

    learn = Learner(
        databunch,
        model,
        opt_func=optimizer,
        loss_func=criterion,
        metrics=[dice, recall_partial, precision_partial, fbeta_score_partial]
    )

    learn.fit_one_cycle(
        epochs,
        lr,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            # SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='best'),
            ShowGraph(learn),
            WandbCallback(learn, log='all')
        ]
    )

    if args.save_model:
        torch.save(model.module.state_dict(), os.path.join(history_path, 'model_weights.pth'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_image_dir", required=True,
                        help="The training image nii directory.")
    parser.add_argument("--train_label_dir", required=True,
                        help="The training label nii directory.")
    parser.add_argument("--val_image_dir", required=True,
                        help="The validation image nii directory.")
    parser.add_argument("--val_label_dir", required=True,
                        help="The validation label nii directory.")
    parser.add_argument("--save_model", default=False,
                        help="Whether to save the trained model.")
    args = parser.parse_args()

    main(args)
