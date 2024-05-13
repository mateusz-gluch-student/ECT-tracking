from dataclasses import dataclass, field
import cv2 
import numpy as np
import os
import random
from typing import Iterable

import ect
from .generator import Generator


@dataclass(slots=True)
class RandomImageGenerator(Generator):
    dirpath: str
    iterlen: int = 100
    img: list[np.ndarray] = field(init=False, default_factory=list)


    def __post_init__(self):
        for root, _, files in os.walk(self.dirpath):
            for file in files:
                self._load_image(os.path.join(root, file))


    def _load_image(self, imgpath: str) -> np.ndarray:
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
 
        # x, y = img.shape[0], img.shape[1]
        # cx, cy = x//2, y//2
        # r = min(cx, cy)
        # dsize = (((np.pi*r)//2)*2, r)

        # filt = ect.sidelobe(
        #     dsize, offset=10, 
        #     flags=ect.ECT_OFFSET_ORIGIN | ect.ECT_GRAYSCALE | ect.ECT_START_NY,
        #     )[:, :, 0]
        
        # img = ect.logpolar_new(img, (cx, cy), dsize, r)
        # img *= filt
        self.img.append(img)
        return img


    def generate(self) -> np.ndarray:
        return random.choice(self.img)


    def images(self) -> Iterable[np.ndarray]:
        for _ in range(self.iterlen):
            yield random.choice(self.img)
    
