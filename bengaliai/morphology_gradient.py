import numpy as np
from albumentations import ImageOnlyTransform
import cv2


class MorphologyGradient(ImageOnlyTransform):
    def __init__(
        self,
        kernel_size: int = 4,
        binarize: bool = False,
        always_apply=False,
        p=0.5,
    ):
        
        super(MorphologyGradient, self).__init__(always_apply, p)
        
        self.kernel_size = kernel_size
        self.binarize = binarize
    
    def apply(self, image, **params):
        if self.binarize:
            image = (image > 0).astype(np.uint8)
            
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)[:, :, np.newaxis]

