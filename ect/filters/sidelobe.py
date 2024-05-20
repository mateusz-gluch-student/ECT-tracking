import numpy as np
import math

from .utils import *
from ..configurators import Config
from ..helpers import vectors

def sidelobe(dsize: tuple[int,int], cfg: Config) -> np.ndarray:
    """Generates sidelobe filter in logpolar domain using
    the following equation

    f(rho,phi) = 1/(1+exp((a-exp(rho)*cos(phi))/slope)) + 1/(1+exp((a+exp(rho)*cos(phi))/slope)) 

    Args:
        shape (tuple): Shape of output array
        radius (float): Max radius of logpolar transform

    Returns:
        np.ndarray: Sidelobe filter for 
    """
    offset = cfg.offset_value_px//2
    slope = cfg.sidelobe_slope

    P, R = dsize
    _, _, xs, _ = vectors((P//2, R), cfg)
    xs = xs[: , R:]
    return sigmoid((xs-offset)/slope) + sigmoid((-xs-offset)/slope)
