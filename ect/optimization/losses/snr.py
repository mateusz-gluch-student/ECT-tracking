import numpy as np  
import cv2

from ect import sidelobe, Config

# def rms(img: np.ndarray) -> float:
#     N: int = img.shape[0]*img.shape[1]
#     sum_sq: np.ndarray = (img*img)/N
#     s: float = sum_sq.sum().sum()
#     return np.sqrt(s/N)

# def rmsn(img: np.ndarray) -> np.ndarray:
#     return img/rms(img)

# def snr(img: np.ndarray, template: np.ndarray) -> float:
#     return -20*np.log10(1/(rms(rmsn(template)-rmsn(img))+1e-9))

def snr(img: np.ndarray, template: np.ndarray) -> float:
    inorm = cv2.normalize(img, None, 1, 0, cv2.NORM_L2)
    tnorm = cv2.normalize(template, None, 1, 0, cv2.NORM_L2)
    rms = np.square(inorm-tnorm).sum().sum()
    return 10*np.log10(rms+1e-12)

def snr_sidelobe(img: np.ndarray, template: np.ndarray) -> float:
    filt = sidelobe(img.shape, Config(offset_value_px=5))
    img -= img.min().min() 
    img *= filt
    template -= template.min().min() 
    template *= filt
    inorm = cv2.normalize(img, None, 1, 0, cv2.NORM_L2)
    tnorm = cv2.normalize(template, None, 1, 0, cv2.NORM_L2)
    rms = np.square(inorm-tnorm).sum().sum()
    return 10*np.log10(rms+1e-12)
