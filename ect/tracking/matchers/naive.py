from dataclasses import dataclass, field

import numpy as np 
import cv2
from numpy import ndarray
from loguru import logger 
from scipy.signal.windows import tukey


from ...helpers import vectors
from ...configurators import Config
from .matcher import Matcher
from ..transformers import Transformer

from icecream import ic
from ect import ilogpolar_new

@dataclass(slots=True)
class NaiveMatcher(Matcher):
    gt: tuple[int, int]
    template_shape: tuple[int, int]
    transformer: Transformer
    tukey_alpha: float = field(default=0.75)
    thresh: float = field(default=0.1)
    template: np.ndarray = field(init=False)
    logpolar: bool = field(default=False)
    do_update_template: bool = field(default=True)    
    offset: int = 5
    
    def initialize(self, image: ndarray) -> np.ndarray:
        #input is not transformed here 
        tpl = self._prepare_window(image, self.template_shape, self.gt)
        
        self.template = self.transformer.transform(tpl)
        return tpl


    def match(self, input: ndarray) -> ndarray:
        #input is not transformed here
        inp = self.transformer.transform(input, center=self.gt)
        xcorr = np.conj(self.template) * inp
        # xcorr = np.conj(inp) * self.template

        xcorr_abs = np.abs(xcorr)
        xcorr_phase = xcorr/xcorr_abs

        xcorr_abs = cv2.normalize(xcorr_abs, None, 0, 1, cv2.NORM_MINMAX)
        thr = np.zeros_like(xcorr)
        thr[xcorr_abs > self.thresh] = 1

        return xcorr_phase * thr

        
    def update(self, input: ndarray, output: ndarray):
        if self.do_update_template:
            self.gt = self._getposition(input, output)
            tpl = self._prepare_window(input, self.template_shape, self.gt)        
            self.template = self.transformer.transform(tpl)
        else:
            tpl = np.zeros_like(input)

        return tpl

    def _prepare_window(
        self, 
        image: np.ndarray, 
        dsize: tuple[int, int], 
        center: tuple[int, int]
    ) -> np.ndarray:
        #input is not transformed here 
        cy, cx = [x//2 for x in image.shape]
        ty, tx = dsize
        gx, gy = center
        # ic(image.shape, dsize, center)
        image_padded = np.pad(image, (ty//2, tx//2), mode="constant", constant_values=0)
        roi = image_padded[gy:gy+ty, gx:gx+tx]
        roi = self._tukeywin(roi)
        tpl = np.zeros_like(image)
        tpl[cy-ty//2:cy+ty//2, cx-tx//2:cx+tx//2] = roi
        return tpl


    def _tukeywin(self, image: np.ndarray) -> np.ndarray:
        Y, X = image.shape
        xwin = tukey(X, self.tukey_alpha)
        ywin = tukey(Y, self.tukey_alpha)
        xfilt, yfilt = np.meshgrid(xwin, ywin)
        
        return image*xfilt*yfilt
    

    def _getposition(self, image: np.ndarray, output: np.ndarray) -> tuple[int, int]:
        Y, X = image.shape
        
        out = np.abs(np.fft.ifft2(output))
        maxidx = np.argmax(out)
        dy, dx = np.unravel_index(maxidx, out.shape)
        OY, OX = out.shape
        ic(dy, dx)

        if self.logpolar:
            
            maxrho = min(X,Y)//2
            dphi = dy/(OY-1)*2*np.pi
            drho = 2*dx/(OX-1)*np.log(maxrho)

            x = int(np.exp(drho)*np.cos(dphi))
            x += self.offset if x > 0 else -self.offset
            y = int(np.exp(drho)*np.sin(dphi))

            x, y = x + self.gt[0], y + self.gt[1]
            x = 0 if x < 0 else (x if x < X else X-1)
            y = 0 if y < 0 else (y if y < Y else Y-1)

            ic(x, y)
            return x, y

        else:
            if dx > OX//2:
                dx -= OX

            if dy > OY//2:
                dy -= OY

            print(f"newpos: x={dx + X//2}, y={dy + Y//2}")
            return dx + X//2, dy + Y//2
