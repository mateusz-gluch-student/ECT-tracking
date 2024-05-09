from dataclasses import dataclass, field
from typing import Iterable
import cv2
import numpy as np

import ect
from .generator import Generator

@dataclass(slots=True)
class SingleImageGenerator(Generator):
    fpath: str
    iterlen: int = 100
    offset: int = 10
    image: np.ndarray = field(init=False)


    def __post_init__(self):
        self.image = cv2.imread(self.fpath)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)/255

        cfg = ect.Config(offset_value_px=self.offset)

        x, y = self.image.shape[0], self.image.shape[1]
        cx, cy = x//2, y//2
        r = min(cx, cy)
        dsize = (314, 100)

        filt = ect.sidelobe(dsize, cfg)        
        self.image = ect.logpolar_new(self.image, (cx, cy), dsize, r, cfg)
        self.image *= filt
    

    def generate(self) -> np.ndarray:
        return self.image
    

    def images(self) -> Iterable[np.ndarray]:
        for _ in range(self.iterlen):
            yield self.image