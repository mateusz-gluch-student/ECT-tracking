import numpy as np

def max_index(image: np.ndarray) -> tuple[float, float]:
    '''
    Calculates position of maximum in an 2D array

    Parameters
    ----------
    image : np.ndarray
        Input Image

    Returns
    -------
    tuple[float, float]
        Coordinates of `max(f(x,y))`
    '''
    max_idx = np.argmax(image)
    max_y, max_x, _ = np.unravel_index(max_idx, image.shape)

    return max_x, max_y
