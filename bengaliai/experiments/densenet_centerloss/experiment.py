"""
Use DenseNet201 from torchvision pretrained models.
Data loading from zip file.
Use albumentation for data augmentation.
Configuration from json file.

Usage:

from bengaliai.experiments import densenet
runner, experiment = densenet.run()
"""

import os
import json
import datetime
import itertools
from collections import OrderedDict

import pandas as pd
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

import catalyst
from catalyst.dl.callbacks import CriterionCallback, CriterionAggregatorCallback
from catalyst.dl import SupervisedExperiment
#from catalyst.dl import SupervisedRunner
from catalyst.dl import SupervisedWandbRunner as SupervisedRunner
from catalyst.dl import utils

from bengaliai.data.zip_dataset import ZIPImageDataset
from bengaliai.models.torchvision_classifier import TorchVisionBengaliClassifier
from bengaliai.metrics import HMacroAveragedRecall, AverageMetric
from bengaliai.data.parquet2zip import parquet_to_images
from bengaliai.config import *

from .center_loss import CenterLoss

use_gpu = torch.cuda.is_available()
experiment_name = 'densenet_central_loss'


class CentralLossTorchVisionBengaliClassifier(TorchVisionBengaliClassifier):
    def forward(self, x):
        if not self.one_channel:
            x = x.repeat(1, 3, 1, 1)
        
        features = x = self.backbone(x)
        x = self.classifier(x)
        
        return [features] + x


class Experiment(SupervisedExperiment):
    def __init__(self):
        super().__init__(
            model=None,
            loaders=None,
            callbacks=[],
            logdir='./logs',
            num_epochs=80,
            main_metric='hmar_avg',
            minimize_metric=False,
            verbose=True,
            monitoring_params={
                "name": experiment_name,
                "tags": ["pytorch", "catalyst", "torchvision", "densenet201"],
                "project": "bengali-ai"
            }
        )

        self._callbacks = OrderedDict((
            # cross entropy
            ('loss_gr', CriterionCallback(
                input_key="grapheme_root",
                output_key="logit_grapheme_root",
                criterion_key='cross_entropy',
                prefix='loss_gr',
            )),
            ('loss_vd', CriterionCallback(
                input_key="vowel_diacritic",
                output_key="logit_vowel_diacritic",
                criterion_key='cross_entropy',
                prefix='loss_vd',
            )),
            ('loss_cd', CriterionCallback(
                input_key="consonant_diacritic",
                output_key="logit_consonant_diacritic",
                criterion_key='cross_entropy',
                prefix='loss_cd',
            )),
            # central loss
            ('central_gr', CriterionCallback(
                input_key="grapheme_root",
                output_key="features",
                criterion_key='central_gr',
                prefix='central_gr',
            )),
            ('central_vd', CriterionCallback(
                input_key="vowel_diacritic",
                output_key="features",
                criterion_key='central_vd',
                prefix='central_vd',
            )),
            ('central_cd', CriterionCallback(
                input_key="consonant_diacritic",
                output_key="features",
                criterion_key='central_cd',
                prefix='central_cd',
            )),
            # aggregator
            ('loss', CriterionAggregatorCallback(
                prefix="loss",
                loss_aggregate_fn="sum",
                loss_keys={
                    "loss_gr": 0.33,
                    "loss_vd": 0.33,
                    "loss_cd": 0.33,
                    "central_gr": 0.1,
                    "central_vd": 0.1,
                    "central_cd": 0.1
                },
            )),
            ('early_stopping', catalyst.dl.EarlyStoppingCallback(4, 'hmar_avg', minimize=False)),
            ('hmar_gr', HMacroAveragedRecall(input_key="grapheme_root", output_key="logit_grapheme_root", prefix="hmar_gr")),
            ('hmar_wd', HMacroAveragedRecall(input_key="vowel_diacritic", output_key="logit_vowel_diacritic", prefix="hmar_wd")),
            ('hmar_cd', HMacroAveragedRecall(input_key="consonant_diacritic", output_key="logit_consonant_diacritic", prefix="hmar_cd")),
            ('hmar_avg', AverageMetric(prefix="hmar_avg", metrics=["hmar_gr", "hmar_wd", "hmar_cd"], weights=[2, 1, 1])),
        ))

        self._criterion = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'central_cd':  CenterLoss(num_classes=7, feat_dim=1920, use_gpu=use_gpu),
            'central_gr':  CenterLoss(num_classes=168, feat_dim=1920, use_gpu=use_gpu),
            'central_vd':  CenterLoss(num_classes=11, feat_dim=1920, use_gpu=use_gpu),
        }

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        if mode == 'train':
            return albumentations.Compose([
                # blur
                albumentations.OneOf([
                    albumentations.Blur((1, 4), p=1.0),
                    albumentations.GaussianBlur(3, p=1.0),
                    albumentations.MedianBlur(blur_limit=5, p=1.0),
                ], p=3/4),
                # transformations
                albumentations.ShiftScaleRotate(scale_limit=0.2, rotate_limit=25, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                # cut and drop
                albumentations.OneOf([
                    albumentations.Cutout(num_holes=10, max_h_size=SIZE//6, max_w_size=SIZE//6, p=1.0),
                    albumentations.CoarseDropout(max_holes=8, max_height=10, max_width=10, p=1.0),
                ], p=2/3),
                # distortion
                albumentations.OneOf([
                    albumentations.OpticalDistortion(0.6, p=1.0),
                    albumentations.GridDistortion(8, 0.06, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                    albumentations.ElasticTransform(sigma=10, alpha=1, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                ], p=3/4),
                # add noise
                albumentations.OneOf([
                    albumentations.GaussNoise((0, 250), p=1.0),
                    albumentations.MultiplicativeNoise(p=1.0),
                ], p=2/3),
                # common
                albumentations.Normalize(TRAIN_MEAN, TRAIN_STD),
                ToTensorV2(),
            ])
        elif mode == 'valid':
            return albumentations.Compose([
                albumentations.Normalize(TRAIN_MEAN, TRAIN_STD),
                ToTensorV2(),
            ])
        else:
            raise ValueError('mode is %s' % mode)
    
    def get_datasets(self, stage: str, **kwargs):
        train_df_ = pd.read_csv('train.csv')
        test_df_ = pd.read_csv('test.csv')
        class_map_df = pd.read_csv('class_map.csv')
        sample_sub_df = pd.read_csv('sample_submission.csv')
        train_df, val_df = train_test_split(train_df_, test_size=VALID_SIZE, random_state=SEED, shuffle=False)

        train_transform = self.get_transforms(stage, 'train')
        train_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, train_df, output_classes.keys(), train_transform, cache_mem=True)
        
        valid_transform = self.get_transforms(stage, 'valid')
        valid_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, val_df, output_classes.keys(), transform=valid_transform, cache_mem=True)
        
        return OrderedDict((
            ('train', train_dataset),
            ('valid', valid_dataset),
        ))

    def get_loaders(self, stage: str, epoch: int = None):
        batch_size = 128
        num_workers = 4
        datasets = self.get_datasets(stage)

        return {name: DataLoader(datasets[name], batch_size=batch_size, num_workers=num_workers, shuffle=True) for name in datasets}
        
    def get_model(self, stage: str):
        model_name = 'densenet201'
        pretrained = True
        backbone = getattr(models, model_name)
        model = CentralLossTorchVisionBengaliClassifier(backbone, output_classes.values(), pretrained)
        return model

    def get_criterion(self, stage: str):
        return self._criterion

    def get_optimizer(self, stage: str, model):
        criterion_params = [c.parameters() for c in self._criterion.values()]

        return torch.optim.Adam([
            {'params': model.parameters(), 'lr': 1e-3},
            {'params': itertools.chain(*criterion_params), 'lr': 1e-5}])


def run(name: str = None, device: str = None, check: bool = False) -> dict:
    device = device or utils.get_device()
    print(f"device: {device}")

    utils.set_global_seed(SEED)

    # convert parquet ot zip
    parquet_to_images(TRAIN, ZIP_TRAIN_FILE, SIZE)
    parquet_to_images(TEST, ZIP_TEST_FILE, SIZE)

    # run experiment
    runner = SupervisedRunner(
        device=device,
        input_key="images",
        output_key=["features"] + ["logit_" + c for c in output_classes.keys()],
        input_target_key=list(output_classes.keys()),)
    experiment = Experiment()
    runner.run_experiment(experiment, check=check)

    return {
        'runner': runner,
        'experiment': experiment,
    }


if __name__ == '__main__':
    run()

