from typing import Iterable

from .spline_filter import splinefilt_rho, splinefilt_phi
from .utils import *
from .weights import *
from ..configurators import Config

def freqnorm(
    dsize: tuple[int, int],
    cfg: Config):

    if len(cfg.freqnorm_knots) == 0:
        cfg.freqnorm_knots = DEFAULT_FNF
 
    return np.exp(splinefilt_rho(dsize, cfg.freqnorm_knots, cfg))


def spacenorm(
   dsize: tuple[int, int],
    cfg: Config):

    if len(cfg.spacenorm_knots) == 0:
        cfg.spacenorm_knots = DEFAULT_SNF
 
    return splinefilt_rho(dsize, cfg.spacenorm_knots, cfg)


def spacenorm_phi(dsize, knot_values: list = DEFAULT_SPHI):
    return splinefilt_phi(dsize, knot_values, len(knot_values))


def freqnorm_phi(dsize, knot_values: list = DEFAULT_FPHI):
    return np.exp(splinefilt_phi(dsize, knot_values, len(knot_values)))
