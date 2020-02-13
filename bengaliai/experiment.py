from .config import *
from .utils import *
from .zip_dataset import *
from .metric_history import *
from .hmar import *
from .torchvision_classifier import *
from .pretrained_classifier import *
from .transforms import *

from torch.utils.data import DataLoader
from torchvision import models

import catalyst
from catalyst.contrib.nn.criterion.focal import FocalLossMultiClass
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, OneCycleLR
from catalyst.dl.callbacks import DiceCallback, IouCallback, CriterionCallback, CriterionAggregatorCallback
from catalyst.dl.runner.supervised import SupervisedRunner
from catalyst.dl.callbacks import CheckpointCallback, MetricCallback
from catalyst.dl import utils

from sklearn.model_selection import train_test_split


def get_model():
    # model = DenseNetMultiLabel(output_classes.values())
    # model = ClippedDenseNet(output_classes.values())
    # model = ClippedDenseNet201(output_classes.values())
    # model = SEResNeXt101(output_classes.values())
    # model = TorchVisionBengaliClassifier(models.densenet121, output_classes.values())
    model = TorchVisionBengaliClassifier(models.densenet201, output_classes.values())

    # checkpoint = torch.load('logs/checkpoints/last.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])

    print('Model params number:', get_n_params(model))

    return model


def get_loaders():
    train_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, train_df, output_classes.keys(), train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    valid_dataset = ZIPImageDataset(ZIP_TRAIN_FILE, val_df, output_classes.keys(), transform=valid_transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    loaders = {
        'train': train_loader,
        'valid': valid_loader,
    }

    return loaders


def get_callbacks():
    callbacks = [
        CriterionCallback(
            input_key="grapheme_root",
            output_key="logit_grapheme_root",
            criterion_key='ce',
            prefix='loss_gr',
        ),
        CriterionCallback(
            input_key="vowel_diacritic",
            output_key="logit_vowel_diacritic",
            criterion_key='ce',
            prefix='loss_wd',
        ),
        CriterionCallback(
            input_key="consonant_diacritic",
            output_key="logit_consonant_diacritic",
            criterion_key='ce',
            prefix='loss_cd',
        ),

        CriterionAggregatorCallback(
            prefix="loss",
            loss_aggregate_fn="sum", # It can be "sum", "weighted_sum" or "mean" in 19.12.1 version
    #         loss_keys=['loss_gr', 'loss_wd', 'loss_cd']
    #         loss_keys={"loss_gr": 2.0, "loss_wd": 1.0, "loss_cd": 1.0},
            loss_keys={"loss_gr": 0.7, "loss_wd": 0.1, "loss_cd": 0.2},
        ),
        catalyst.dl.EarlyStoppingCallback(3, 'hmar', minimize=False),
        # metrics
        HMacroAveragedRecall(),
        hhmar,
    ]

    return callbacks


def get_scheduler():
    # scheduler = None
    scheduler = StepLR(optimizer, 3, 0.75)
    # scheduler = CosineAnnealingLR(optimizer, 3, 1e-6)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', 0.7, 3, min_lr=1e-10)
    # scheduler = CyclicLR(optimizer, 1e-6, 1e-2, 500)
    # scheduler = OneCycleLR(optimizer, 1e-1, int(1e4))

    return scheduler

train_df_ = pd.read_csv('train.csv')
test_df_ = pd.read_csv('test.csv')
class_map_df = pd.read_csv('class_map.csv')
sample_sub_df = pd.read_csv('sample_submission.csv')

train_df, val_df = train_test_split(train_df_, test_size=0.1, random_state=42, shuffle=False)

model = get_model()
loaders = get_loaders()

criterion = {
#     "ce": nn.CrossEntropyLoss(),
    "ce": FocalLossMultiClass(),
}

runner = SupervisedRunner(
    device=device,
    input_key="images",
    output_key=["logit_" + c for c in output_classes.keys()],
    input_target_key=list(output_classes.keys()),
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = get_scheduler()

hhmar = MetricHistory('hmar')
callbacks = get_callbacks()

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    logdir=logdir,
    num_epochs=epochs,
#     main_metric="hmar",
    minimize_metric=False,
    fp16=None,
    monitoring_params=None,
    verbose=True,
#     check=True,
#     resume='logs/checkpoints/train.1.pth',  # path to checkpoint for model
#     state_kwargs: Dict = None,  # additional state params to ``State``
)
