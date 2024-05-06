import numpy as np

from ..configurators import Config

def pad_coords(coords: np.ndarray, bounds: tuple[int, int], replace: int = 0) -> np.ndarray:
    coords[coords > bounds[1]] = replace
    coords[coords < bounds[0]] = replace
    return coords


def floor_sample(
    image: np.ndarray, 
    xfloats: np.ndarray, 
    yfloats: np.ndarray
) -> np.ndarray:  
    #padding
    xfloats = pad_coords(xfloats, (0, image.shape[1]))
    yfloats = pad_coords(yfloats, (0, image.shape[0]))
    image[0, 0] = 0

    y_coords = yfloats.astype(int)
    x_coords = xfloats.astype(int)

    #without bilinear map
    out = image[y_coords, x_coords]
    return out


def bilinear_sample(
    image: np.ndarray, 
    xfloats: np.ndarray, 
    yfloats: np.ndarray
) -> np.ndarray:
    pv = 3
    overlap = 1

    xfloats += pv
    yfloats += pv

    from icecream import ic
    ymax, xmax = image.shape
    image = np.pad(image, pv, mode="reflect")

    xfloors = np.floor(pad_coords(xfloats, (pv-overlap, xmax+pv+overlap, 0))).astype(int)
    xceils = np.ceil(pad_coords(xfloats, (pv-overlap, xmax+pv+overlap), 0)).astype(int)
    yfloors = np.floor(pad_coords(yfloats, (pv-overlap, ymax+pv+overlap), 0)).astype(int)
    yceils = np.ceil(pad_coords(yfloats, (pv-overlap, ymax+pv+overlap), 0)).astype(int)
    image[0, 0] = 0

    out_yceils = (xceils-xfloats)*image[yceils, xfloors] + (xfloats-xfloors)*image[yceils, xceils]
    from matplotlib import pyplot as plt
    out_yfloors = (xceils-xfloats)*image[yfloors, xfloors] + (xfloats-xfloors)*image[yfloors, xceils]
    from matplotlib import pyplot as plt
    out = (yceils-yfloats)*out_yfloors + (yfloats-yfloors)*out_yceils 
    return out
