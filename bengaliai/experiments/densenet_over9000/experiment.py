"""
Use DenseNet201 from torchvision pretrained models.
Data loading from zip file.
Use albumentation for data augmentation.
Configuration from json file.

Usage:

from bengaliai.experiments import densenet
runner, experiment, config = densenet.run()
"""

import os
import json
import datetime
from collections import OrderedDict

import pandas as pd
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split

from torchvision import models

from catalyst.dl import ConfigExperiment
#from catalyst.dl import SupervisedRunner
from catalyst.dl import SupervisedWandbRunner as SupervisedRunner
from catalyst.dl import utils

from bengaliai.data.zip_dataset import ZIPImageDataset
from bengaliai.models.torchvision_classifier import TorchVisionBengaliClassifier
from bengaliai.metrics import HMacroAveragedRecall, AverageMetric
from bengaliai.data.parquet2zip import parquet_to_images
from bengaliai.config import *
from .config import experiment_name, experiment_config

import wandb

from catalyst.core.registry import OPTIMIZERS
from bengaliai import over9000
OPTIMIZERS.add(over9000.Over9000)


class Experiment(ConfigExperiment):
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        if mode == 'train':
            return albumentations.Compose([
                # blur
                albumentations.Blur((1, 2), p=1.0),
                # transformations
                albumentations.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                # cut and drop
                albumentations.Cutout(num_holes=8, max_h_size=SIZE//8, max_w_size=SIZE//8, p=1.0),
                # distortion
                albumentations.OpticalDistortion(0.3, p=1.0),
                albumentations.GridDistortion(5, 0.03, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
                # add noise
                albumentations.GaussNoise((0, 150), p=1.0),
                albumentations.MultiplicativeNoise(p=1.0),
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
        train_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, train_df, output_classes.keys(), train_transform)
        
        valid_transform = self.get_transforms(stage, 'valid')
        valid_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, val_df, output_classes.keys(), transform=valid_transform)
        
        return OrderedDict((
            ('train', train_dataset),
            ('valid', valid_dataset),
        ))
    
    @staticmethod
    def _get_model(model_name: str, output_classes: list, pretrained: bool):
        backbone = getattr(models, model_name)
        model = TorchVisionBengaliClassifier(backbone, output_classes, pretrained)
        return model


def load_config_from_json(filepath: str = __file__):
    path = os.path.dirname(os.path.abspath(filepath))

    with open(os.path.join(path, 'config.json')) as f:
        return json.load(f)


def run(name: str = None, config: dict = None, device: str = None) -> dict:
    config = config or experiment_config
    device = device or utils.get_device()
    print(f"device: {device}")

    utils.set_global_seed(SEED)

    # inititalize weigths & biases
    name = name or '_'.join(filter(None, [experiment_name, f"{datetime.datetime.now():%Y-%m-%d-%S}"]))
    wandb.init(name, project=WANDB_PROJECT, id=name)

    # convert parquet ot zip
    parquet_to_images(TRAIN, ZIP_TRAIN_FILE, SIZE)
    parquet_to_images(TEST, ZIP_TEST_FILE, SIZE)

    # run experiment
    runner = SupervisedRunner(
        device=device,
        input_key="images",
        output_key=["logit_" + c for c in output_classes.keys()],
        input_target_key=list(output_classes.keys()),)
    experiment = Experiment(config)
    runner.run_experiment(experiment)

    return {
        'runner': runner,
        'experiment': experiment,
        'config': config,
    }


if __name__ == '__main__':
    run()

