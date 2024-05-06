import numpy as np
import cv2

def square_grid(
    dsize: tuple[int, int], 
    step: int, 
    thickness: int = 1
) -> np.ndarray:
    '''
    Generates square grid of size [dsize]
    with parametrized step.

    Outputs grid as grayscale image.
    '''
    out: np.ndarray = np.ones(dsize, dtype=float)

    # x grid
    for x in range(0, dsize[1]+1, step):
        # ic(x)
        out = cv2.line(out, (x, 0), (x, dsize[0]), (0), thickness)
    
    # y grid
    for y in range(0, dsize[0]+1, step):
        # ic(y)
        out = cv2.line(out, (0, y), (dsize[1], y), (0), thickness)
    
    return out

def circular_grid(
    dsize: tuple[int, int], 
    radius_step: int, 
    angle_step_deg: int,
    thickness: int = 1
) -> np.ndarray:
    '''
    Generates circular grid of size [dsize]
    centered at middle of the image and
    parametrized radius and angle step.

    Outputs grid as grayscale image.
    '''
    out: np.ndarray = np.ones(dsize, dtype=float)
    xc, yc = dsize[0]//2, dsize[1]//2

    max_radius = max(xc, yc)

    for r in range(0, max_radius+1, radius_step):
        # ic(xc, yc, r)
        out = cv2.circle(out, (xc, yc), r, (0), thickness)
    
    for ang in range(0, 360, angle_step_deg):
        phi: float = np.pi * ang / 180
        px: int = int(xc + max_radius*np.cos(phi))
        py: int = int(yc + max_radius*np.sin(phi))
        # ic((xc, yc), (px, py), ang)
        out = cv2.line(out, (xc, yc), (px, py), (0), thickness)
    
    return out