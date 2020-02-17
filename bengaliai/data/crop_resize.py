import numpy as np
import cv2
from .bbox import bbox


def crop_resize(img, size=128, pad=16):
    h, w, *_ = img.shape
    ymin, ymax, xmin, xmax = bbox(img[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < w - 13) else w
    ymax = ymax + 10 if (ymax < h - 10) else h
    img = img[ymin:ymax, xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))


