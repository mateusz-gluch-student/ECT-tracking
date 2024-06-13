import numpy as np

from ..configurators import Config


def vectors(shape: tuple[int, int], cfg: Config
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Generates base vectors
    for a given shape, in range 
    [1, log(R)] x (0, pi)

    These vectors apply to image folded onto one
    half of plane (x > 0)

    Parameters
    ----------
    shape : tuple[int, int]
        shape of kernel image
    flags : int, optional
        launch configuration, by default ECT_START_PX

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple of gamma, phi, x and y vectors
    '''

    P, R = shape

    gamma = np.linspace(-1, 1, 2*R, endpoint=False) * np.log(R)
    phi = np.linspace(0 ,1, P, endpoint=False) * 2*np.pi
    phi -= cfg.start_angle_rad

    gammas, phis = np.meshgrid(gamma, phi)
    xs = np.exp(gammas) * np.cos(phis)    
    ys = np.exp(gammas) * np.sin(phis)

    return gammas, phis, xs, ys