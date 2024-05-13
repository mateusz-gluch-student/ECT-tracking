import cv2
import math
import numpy as np
from loguru import logger
from icecream import ic

from .utils_new import *
from ..configurators import Config

DEFAULT_CONFIG = Config(
    mode = "offset",
    interpolation = "bilinear",
    start_angle_deg = 90,
    offset_value_px = 10
)


def ilogpolar(
    image: np.ndarray, 
    dsize: tuple[int, int] = None,  
    radius: int = None,
    cfg: Config = DEFAULT_CONFIG
    ) -> cv2.Mat:
    """Performs inverse logarithmic polar mapping on a source image. 

    Args:
        img (cv2.Mat): source image
        dsize (tuple[int, int]): destination image shape
        center (tuple[int, int]): center of transformed region
        radius (int): radius of transformed region
        offset (int): origin offset, required in ECT_OFFSET_ORIGIN
        flags (int, optional): execution flags. Defaults to INTER_NONE.

    Returns:
        cv2.Mat: Inverse polar mapped source image
    """

    # if len(img.shape) == 2:
    #     img = img[:,:,np.newaxis]

    # get radius 
    # if radius is None or radius <= 0:
    #     radius = img.shape[1]

    # get dsize
    # if dsize is None or dsize[1] <= 0 or dsize[0] <= 0:
    #     out_width = round(2*radius)
    #     out_height = round(2*radius)
    # else:
    #     out_width = round(dsize[1])
    #     out_height = round(dsize[0])

    # get center
    # if center is None or center[0] <= 0 or center[1] <= 0:
    #     cx = cy = round(radius)
    # else:
    #     cx = round(center[0])
    #     cy = round(center[1])

    x_range: np.ndarray = np.linspace(-radius, radius, dsize[1])
    y_range: np.ndarray = np.linspace(-radius, radius, dsize[0])

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    if cfg.mode == "omit":
        logger.debug("Running logpolar transform in omit mode")
        rho_grid: np.ndarray = np.log(x_grid*x_grid + y_grid*y_grid + 1e-6)/2
        phi_grid: np.ndarray = np.arctan2(y_grid, x_grid) + cfg.start_angle_rad

    elif cfg.mode == "opencv":
        logger.debug("Running logpolar transform in opencv mode")
        blind = cfg.offset_value_px

        rho_grid = np.log(x_grid*x_grid + y_grid*y_grid + blind)/2
        phi_grid = np.arctan2(y_grid, x_grid) + cfg.start_angle_rad

    else:
        logger.debug("Running logpolar transform in offset mode")
        offset_bool: np.ndarray = (x_grid > 0).astype(int) 
        offset: np.ndarray = (offset_bool*2 - 1) * cfg.offset_value_px
        x_grid -= offset

        rho_grid = np.log(x_grid*x_grid + y_grid*y_grid)/2
        phi_grid = np.arctan2(y_grid, x_grid) + cfg.start_angle_rad
        
    rho_grid *= (image.shape[1]-1)/np.log(radius)
    phi_grid[phi_grid <= 0] += 2*np.pi
    phi_grid *= (image.shape[0]-1)/2/np.pi

    if cfg.interpolation == "none":
        logger.debug("Applying floor px mapping")
        return floor_sample(image, rho_grid, phi_grid)
    else:
        logger.debug("Applying bilinear pixel mapping")
        return bilinear_sample(image, rho_grid, phi_grid)

