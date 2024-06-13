import numpy as np
import cv2

from .utils_new import bilinear_sample, floor_sample
from ..configurators import Config
from ..helpers import vectors

from loguru import logger 

DEFAULT_CONFIG = Config(
    mode = "offset",
    interpolation = "bilinear",
    start_angle_deg = 90,
    offset_value_px = 10
)

def logpolar(
    image: np.ndarray, 
    center: tuple[int, int], 
    dsize: tuple[int, int] = (0, 0), 
    radius: int = 0, 
    cfg: Config = DEFAULT_CONFIG
) -> np.ndarray:

    xc, yc = center

    phi_range: np.ndarray = np.linspace(1e-9, 2, dsize[0], endpoint=False) * np.pi
    phi_range -= cfg.start_angle_rad

    r = np.log(radius)
    rho_range: np.ndarray = np.linspace(1e-9, 1, dsize[1], endpoint=False) * r

    rho_grid, phi_grid = np.meshgrid(rho_range, phi_range)

    # P, R = dsize
    # rho_grid, phi_grid, _, _ = vectors((P//2, R), cfg)
    # rho_grid = rho_grid[:, R:]
    # phi_grid = phi_grid[:, R:]

    if cfg.mode == "omit":
        logger.debug("Running logpolar transform in omit mode")
        x_grid = np.exp(rho_grid) * np.cos(phi_grid) + xc
        y_grid = np.exp(rho_grid) * np.sin(phi_grid) + yc
    elif cfg.mode == "opencv":
        logger.debug(f"Running logpolar transform in opencv mode blind_zone={cfg.offset_value_px}")
        blind = cfg.offset_value_px
        x_grid = (np.exp(rho_grid)-blind) * np.cos(phi_grid) + xc
        y_grid = (np.exp(rho_grid)-blind) * np.sin(phi_grid) + yc
    else:
        logger.debug(f"Running logpolar transform in offset mode offset={cfg.offset_value_px}")
        x_grid = np.exp(rho_grid) * np.cos(phi_grid) + xc# + cfg.offset_value_px

        offset_bool: np.ndarray = (np.cos(phi_grid) > 0).astype(int) 
        offset: np.ndarray = (offset_bool*2 - 1) * cfg.offset_value_px
        x_grid = x_grid + offset
        y_grid = np.exp(rho_grid) * np.sin(phi_grid) + yc

    if cfg.interpolation == "none":
        logger.debug("Applying floor px mapping")
        return floor_sample(image, x_grid, y_grid)
    else:
        logger.debug("Applying bilinear pixel mapping")
        return bilinear_sample(image, x_grid, y_grid)

