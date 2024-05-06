import numpy as np
import cv2

from ect import sidelobe

def l1dist(img: np.ndarray, template: np.ndarray) -> float:
    inorm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    tnorm = cv2.normalize(template, None, 0, 1, cv2.NORM_MINMAX)

    dist = np.abs(inorm-tnorm)
    return dist.sum().sum()


def l2dist(img: np.ndarray, template: np.ndarray) -> float:
    inorm = cv2.normalize(img, None, 0, 1, cv2.NORM_L2)
    tnorm = cv2.normalize(template, None, 0, 1, cv2.NORM_L2)
    return np.square(inorm-tnorm).sum().sum()


def l2dist_sidelobe(img: np.ndarray, template: np.ndarray) -> float:
    filt = sidelobe(img.shape, offset=5)
    inorm = cv2.normalize(img*filt, None, 0, 1, cv2.NORM_L2)
    tnorm = cv2.normalize(template*filt, None, 0, 1, cv2.NORM_L2)

    dist1d = inorm-tnorm
    return np.sqrt((dist1d*dist1d).sum().sum())
    