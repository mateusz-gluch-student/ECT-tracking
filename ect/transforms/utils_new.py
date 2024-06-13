## new utils

import numpy as np

from ..configurators import Config, AntialiasParameters
from ..filters import sigmoid
from ..helpers import vectors

# def fold_logpolar(image: np.ndarray) -> np.ndarray:
#     phi, RHO = image.shape[:2]
#     if phi % 2:
#         out = image[:phi//2+1, :] + image[phi//2:, :]*1j
#     else:    
#         out = image[:phi//2, :] + image[phi//2:, :]*1j
#     return out
    

# def unfold_logpolar(image: np.ndarray) -> np.ndarray:
#     return np.vstack([np.real(image), np.imag(image)])


def xcorr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Calculates 2D cross-correlation between two ndarrays
    using following equation:

    xcorr(a, b) = F(A)' * F(B)

    where F stands for Fourier transform
    
    Parameters
    ----------
    A, B : np.ndarray
        Input arrays

    Returns
    -------
    np.ndarray
        Cross-correlation between two arrays
    '''
    A_t = np.fft.fft2(A, axes=(0, 1))
    B_t = np.fft.fft2(B, axes=(0, 1))
    out_t = np.conjugate(A_t) * B_t
    return np.fft.ifft2(out_t, axes=(0,1))


# def vectors(shape: tuple[int, int], cfg: Config
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     '''
#     Generates base vectors
#     for a given shape, in range 
#     [1, log(R)] x (0, pi)

#     These vectors apply to image folded onto one
#     half of plane (x > 0)

#     Parameters
#     ----------
#     shape : tuple[int, int]
#         shape of kernel image
#     flags : int, optional
#         launch configuration, by default ECT_START_PX

#     Returns
#     -------
#     tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
#         Tuple of gamma, phi, x and y vectors
#     '''
#     P, R = shape

#     gamma = np.linspace(-1+1/R, 1, 2*R) * np.log(R)
#     phi = np.linspace(1/P, 2, 2*P) * np.pi
#     phi -= cfg.start_angle_rad

#     gammas, phis = np.meshgrid(gamma, phi)
#     xs = np.exp(gammas) * np.cos(phis)    
#     ys = np.exp(gammas) * np.sin(phis)

#     return gammas, phis, xs, ys

def antialias(
    kernel: np.ndarray, 
    params: list[AntialiasParameters],
    ) -> np.ndarray:
    '''
    Applies antialiasing filter to a kernel

    Parameters
    ----------
    kernel : np.ndarray
        _description_
    cfg: Config
        ECT library launch configuration
        
    Returns
    -------
    np.ndarray
        Kernel with applied antialiasing
    '''
    for p in params:  
        filt = sigmoid(1/p.slope*(p.threshold-p.factor*abs(p.vector)))
        kernel *= filt

    return kernel


def mod_image(
    image: np.ndarray, 
    cfg: Config):
    '''
    Prepares imadd

    Parameters
    ----------
    image : np.ndarray
        _description_
    ect_offset : int
        _description_
    flags : int, optional
        _description_, by default ECT_OMIT_ORIGIN | ECT_START_PX

    Returns
    -------
    _type_
        _description_
    '''
    if cfg.transform == "ect":
        offset = cfg.offset_value_px
        other_offset = cfg.ect_offset_value_px
        ect_factor = 1
    elif cfg.transform == "iect":
        offset = cfg.ect_offset_value_px
        other_offset = cfg.offset_value_px
        ect_factor = -1

    P, R = image.shape[:2]
    image_padded = np.zeros((P, 2*R), dtype=complex)

    rhos, _, xs, _ = vectors((P, R), cfg)

    rhos = rhos[:P, R:]
    xs = xs[:P, R:] 

    offset_bool: np.ndarray = (xs > 0).astype(int) 
    off: np.ndarray = (offset_bool*2 - 1) * offset
    xs += off

    if cfg.mode == "offset":
        image_padded[:, :R] = np.conjugate(image) * \
            np.exp(2*rhos - ect_factor*2*np.pi*1j*other_offset*xs/R)
    elif cfg.mode == "omit":
        image_padded[:, :R] = np.conjugate(image) * np.exp(2*rhos)
    else:
        raise AttributeError('logpolar mode not in ["offset", "omit"]')

    return image_padded


def shift(image: np.ndarray, cfg: Config) -> np.ndarray:

    if cfg.transform == "ect":
        offset = cfg.offset_value_px
        other_offset = cfg.ect_offset_value_px
        ect_factor = 1
    elif cfg.transform == "iect":
        offset = cfg.ect_offset_value_px
        other_offset = cfg.offset_value_px
        ect_factor = -1

    P, R = image.shape[:2]
    _, _, xs, _ = vectors((P, R), cfg)
    xs -= other_offset
    xs = xs[:P, R:]
    # xs = xs[:, :R]


    return np.exp(ect_factor*2*np.pi*1j*offset*xs/R)