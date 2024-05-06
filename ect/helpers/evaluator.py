# imports
import ect
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

from dataclasses import dataclass, field
from typing import Optional

RADIUS = 200

@dataclass
class ECTEvaluator:
    '''Evaluator object'''
    img_path: str
    r: float = RADIUS
    a: float = RADIUS/20
    b: float = RADIUS/10
    img: np.ndarray = field(init=False)
    inv: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        src = cv2.imread(self.img_path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        self.img = ect.logpolar(src, self.r, offset=self.a)
        sidelobe = ect.sidelobe(self.img.shape[:2], offset=self.a)
        self.img = self.img * sidelobe


    def transform(self):
        '''Transform (and immediately invert) image using ECT'''

        dsize = self.img.shape[:2]

        snf = ect.spacenorm(dsize, self.r)
        fnf = ect.freqnorm(dsize, self.r)
        # ang = ect.angular_filter(dsize)

        ect_img = ect.fect(self.img, self.a, self.b)
        ect_img = fnf * ect_img
        self.inv = ect.ifect(ect_img, self.a, self.b)
        self.inv = snf * self.inv

        return self.inv


    def eval(self, eval_func: Callable[[np.ndarray, np.ndarray], float]):
        '''
        Evaluate (using a eval function) a 
        difference between image and its transform
        '''
        if self.inv is None:
            self.inv = self.transform()

        result = eval_func(self.img, self.inv)

        print(f"{eval_func.__name__} = {result:.2f}")


    def show(
        self,
        norm_fcn: Callable[[np.ndarray], np.ndarray],
        *params
    ):
        '''Show result of evaluation in logpolar domain'''

        if self.inv is None:
            self.inv = self.transform()

        log_img = norm_fcn(self.img)
        inv_img = norm_fcn(self.inv)

        diff = log_img - inv_img

        plt.figure(figsize = (20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(ect.complex_to_hsv(self.img))

        plt.subplot(1, 3, 2)
        plt.imshow(ect.complex_to_hsv(inv_img))

        plt.subplot(1, 3, 3)
        plt.imshow(ect.complex_to_hsv(diff))


    def show_cart(
        self,
        norm_fcn: Callable[[np.ndarray], np.ndarray],
        *params
    ):
        '''Show result of evaluation in cartesian domain'''
        
        if self.inv is None:
            self.inv = self.transform()

        
        log_img = norm_fcn(self.img)
        inv_img = norm_fcn(self.inv)

        diff = ect.complex_to_hsv(log_img - inv_img)
        log_img = ect.complex_to_hsv(log_img)
        inv_img = ect.complex_to_hsv(inv_img)

        diff_cart = ect.ilogpolar(diff, RADIUS = self.r, offset=self.a)
        src_cart = ect.ilogpolar(log_img, RADIUS = self.r, offset=self.a)
        inv_cart = ect.ilogpolar(inv_img, RADIUS = self.r, offset=self.a)

        plt.figure(figsize = (20, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(src_cart)

        plt.subplot(1, 3, 2)
        plt.imshow(inv_cart)

        plt.subplot(1, 3, 3)
        plt.imshow(diff_cart)
