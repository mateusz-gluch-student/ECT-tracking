import cv2
import numpy as np

from ..configurators import Config, AntialiasParameters
from .utils_new import (
    antialias, 
    mod_image, 
    xcorr, 
    shift,
    fold_logpolar,
    unfold_logpolar)

from ..helpers import vectors

def transform(inp: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.transform == "ect":
        ect_factor = 1
    elif cfg.transform == "iect":
        ect_factor = -1

    P, R = inp.shape[:2]
    _, _, xs, ys = vectors((P, R), cfg)
    kernel = np.exp(2 * np.pi * 1j * xs * ect_factor)
    
    if cfg.antialias:
        cfg.antialias_params = _antialias(xs, ys, (P, R), cfg.antialias_factors)
        kernel = antialias(kernel, cfg.antialias_params)

    image_padded = mod_image(inp, cfg)
    out = xcorr(image_padded, kernel)
    out = out[:P, :R]
    return shift(inp, cfg) * out if cfg.mode == "offset" else out


def _antialias(xs, ys, dsize, factors) -> list[AntialiasParameters]:
    return [
        AntialiasParameters(factor=factors[0], threshold=np.log(dsize[1]), vector=xs),
        AntialiasParameters(factor=factors[1], threshold=np.pi, vector=ys)
    ]


def fect(
    image: cv2.Mat | np.ndarray,
    cfg: Config
    ) -> cv2.Mat:
    '''
    Implementation of Fast ECT O(n^2*logn)
    '''
    cfg.validate()
    cfg.transform = "ect"
    # image = fold_logpolar(image)
    P, _ = image.shape
    return transform(image, cfg)
    out_right = transform(image[:P//2, :], cfg)
    out_left = transform(image[P//2:, :], cfg)
    return np.vstack([out_right, out_left])

def ifect(
    ect: cv2.Mat,
    cfg: Config
) -> cv2.Mat:
    '''
    Implementation of Inverse FECT O(n^2)
    '''
    cfg.validate()
    cfg.transform = "iect"
    return transform(ect, cfg)
    P, _ = ect.shape
    out_right = transform(ect[:P//2, :], cfg)
    # out_right = cv2.normalize(out_right, None, 1, 0, cv2.NORM_MINMAX)
    out_left = transform(ect[P//2:, :], cfg)
    # out_left = cv2.normalize(out_right, None, 1, 0, cv2.NORM_MINMAX)
    return np.vstack([out_right, out_left])
