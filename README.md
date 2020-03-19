# Kaggle Bengali.AI Competition Experiments & Utils

## Install

There is no PyPI repo. Install from github:

```
pip install -U git+https://github.com/ihoromi4/bengali-ai.git
```

## Dependencies

* pytorch
* torchvision
* pretrainedmodels
* catalyst
* albumentations
* efficientnet_pytorch
* iterative-stratification
* wandb

## Attention

I used [W&B](www.wandb.com) service for logging and experiment analysis. So you need W&B account to run experiments.

## Usage

1. Download competition dataset (bash):

```
pip install kaggle
kaggle competitions download -c bengaliai-cv19
unzip -u bengaliai-cv19.zip
```

2. Run experiment (python):

```
from bengaliai.experiments.seresnext50_deephead import experiment

experiment.run(check=False)
```
