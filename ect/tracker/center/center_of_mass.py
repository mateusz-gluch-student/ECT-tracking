import numpy as np
import cv2
import ect

def center_of_mass(image: np.ndarray) -> tuple[float, float]:
    '''
    Calculates Center of Mass of an Image according to equation:

    `x_mean = sum(f(x,y)*x)/X`
    `y_mean = sum(f(x,y)*y)/Y`

    In order to provide better accuracy, the image
    is normalized before calculation and thresholded at 0.5

    Parameters
    ----------
    image : np.ndarray
        Input Image

    Returns
    -------
    tuple[float, float]
        Coordinates of center of mass
    '''
    image = image[:, :, 0]
    image = ect.norm_minmax(image, 0, 1, dtype=np.float64)
    image[image < 0.5] = 0
    moments = cv2.moments(image)

    c0 = moments["m00"]
    cx = moments["m10"]
    cy = moments["m01"]

    return cx/c0, cy/c0 