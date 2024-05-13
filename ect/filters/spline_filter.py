from scipy.interpolate import CubicSpline
import numpy as np

from .utils import vector_gen
from ..helpers import vectors
from ..configurators import Config


def splinefilt_rho(dsize: tuple[int, int], knots: list[float], cfg: Config) -> np.ndarray:
    num_knots = len(knots)
    P, R = dsize
    if P % 2:
        rhos, _, _, _ = vectors((P//2+1, R), cfg)
    else:
        rhos, _, _, _ = vectors((P//2, R), cfg)
    rhos = rhos[:P, R:]
    max_rho = rhos[-1, -1]
    min_rho = rhos[0, 0]

    x_knots = np.linspace(min_rho, max_rho, num_knots)
    # print(x_knots)
    polyfilt = CubicSpline(x=x_knots, y=knots, bc_type='natural')

    return polyfilt(rhos, extrapolate=True)


def splinefilt_phi(dsize: tuple[int, int], knots: list[float], cfg: Config) -> np.ndarray:
    num_knots = len(knots)
    x_knots = np.linspace(0, dsize[0], num_knots)
    polyfilt = CubicSpline(x=x_knots, y=knots, bc_type='natural')
    P, R = dsize
    _, phis, _, _ = vectors((P//2, R), cfg)
    phis = phis[:, R:]

    return polyfilt(phis)
