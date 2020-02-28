"""
Install in development mode:
python3 setup.py develop
or
pip install -e .[lib]
"""

import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="bengali-ai",
    version="0.0.1",
    author="Ihor Omelchenko",
    author_email="counter3d@gmail.com",
    description="",
    # license="",
    keywords="kaggle bengali-ai",
    packages=find_packages(exclude=["tests"]),
    # long_description=read('README'),
    classifiers=[
        'Topic :: Utilities',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'pyarrow',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'torch',
        'torchvision',
        'catalyst[contrib]',
        'albumentations>=0.4.3',
        'opencv-contrib-python',
        'tqdm',
        'wandb',
        'pretrainedmodels',
        'efficientnet_pytorch',
        'kaggle',
    ],
)

