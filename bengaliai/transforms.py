import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from .config import *

train_transform = albumentations.Compose([
#     albumentations.Resize(SIZE, SIZE),
    # blur
    albumentations.Blur((1, 2), p=1.0),
#     albumentations.GaussianBlur(3, p=1.0),
#     albumentations.GlassBlur(p=1.0),
#     albumentations.MedianBlur(blur_limit=5, p=1.0),
#     albumentations.MotionBlur(p=1.0),
#     albumentations.RandomBrightnessContrast(p=1.0),
    # transformations
#     albumentations.Rotate(30)
    albumentations.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
#     albumentations.RandomResizedCrop(128, 128, scale=(0.5, 1.1)),
    # cut and drop
    albumentations.Cutout(num_holes=8, max_h_size=SIZE//8, max_w_size=SIZE//8, p=1.0),
#     albumentations.CoarseDropout(max_holes=8, max_height=10, max_width=10, p=1.0),
#     albumentations.GridDropout(),
    # distortion
    albumentations.OpticalDistortion(0.3, p=1.0),
    albumentations.GridDistortion(5, 0.03, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
#     albumentations.ElasticTransform(sigma=10, alpha=1, alpha_affine=10, p=1.0),  # x2 transfrom time
#     albumentations.RandomGridShuffle(),
    # add noise
    albumentations.GaussNoise((0, 150), p=1.0),
    albumentations.MultiplicativeNoise(p=1.0),
#     Binarize(),
#     albumentations.Equalize(),
    albumentations.Normalize(TRAIN_MEAN, TRAIN_STD),
    ToTensorV2(),
])

valid_transform = albumentations.Compose([
    albumentations.Normalize(TRAIN_MEAN, TRAIN_STD),
    ToTensorV2(),
])

