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
from os.path import basename, dirname, abspath
import json
import datetime
from collections import OrderedDict

import pandas as pd
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split

import torch
from torchvision import models

from catalyst.dl import ConfigExperiment
from catalyst.dl import SupervisedRunner
from catalyst.dl import SupervisedWandbRunner
from catalyst.dl import utils

from bengaliai.data.zip_dataset import ZIPImageDataset
from bengaliai.models.pretrained_classifier import PretrainedModelsBengaliClassifier
from bengaliai.metrics import HMacroAveragedRecall, AverageMetric
from bengaliai.data.parquet2zip import parquet_to_images
from bengaliai.config import *
from .config import experiment_config
from ...gridmask import GridMask

SIZE = 128
ZIP_TRAIN_FILE = f'train{SIZE}.zip'
ZIP_TEST_FILE = f'test{SIZE}.zip'
EXPERIMENT_NAME = basename(dirname(abspath(__file__)))


class Experiment(ConfigExperiment):
    def __init__(self, config, model_filepath: str = None):
        super().__init__(config)

        self._model_filepath = model_filepath

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
                # distortion
                albumentations.OneOf([
                    albumentations.OpticalDistortion(1.2, p=1.0),
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
                GridMask(5, rotate=45, p=0.9),
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
    
    def _get_model(self, model_name: str, output_classes: list, pretrained: str):
        model = PretrainedModelsBengaliClassifier(model_name, output_classes, pretrained)

        if self._model_filepath:
            checkpoint = torch.load(self._model_filepath)
            model.load_state_dict(checkpoint['model_state_dict'])

        return model


def load_config_from_json(filepath: str = __file__):
    path = os.path.dirname(os.path.abspath(filepath))

    with open(os.path.join(path, 'config.json')) as f:
        return json.load(f)


def run(
        name: str = None,
        config: dict = None,
        model_filepath: str = None,
        device: str = None,
        check: bool = False) -> dict:

    config = config or experiment_config
    device = device or utils.get_device()
    print(f"device: {device}")

    utils.set_global_seed(SEED)

    config['monitoring_params']['name'] = EXPERIMENT_NAME
    config['stages']['state_params']['checkpoint_data']['image_size'] = SIZE

    # convert parquet ot zip
    parquet_to_images(TRAIN, ZIP_TRAIN_FILE, SIZE)
    parquet_to_images(TEST, ZIP_TEST_FILE, SIZE)

    # run experiment
    RunnerClass = SupervisedRunner if check else SupervisedWandbRunner
    runner = RunnerClass(
        device=device,
        input_key="images",
        output_key=["logit_" + c for c in output_classes.keys()],
        input_target_key=list(output_classes.keys()),)
    experiment = Experiment(config, model_filepath)
    runner.run_experiment(experiment, check=check)

    return {
        'runner': runner,
        'experiment': experiment,
        'config': config,
    }


if __name__ == '__main__':
    run()

