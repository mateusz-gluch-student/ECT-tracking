from dataclasses import dataclass, field
from typing import Iterable
import cv2
import numpy as np

import random

import ect
from .generator import Generator
from ..samples import sine_unimodal, Mode

@dataclass(slots=True)
class UnimodalGenerator(Generator):
    mode: Mode
    iterlen: int = 100
    offset: int = 10
    dsize: tuple[int, int] = (314, 100)
    image: np.ndarray = field(init=False)
    logpolar: bool = field(default=True)

    def __post_init__(self):
        self.image = sine_unimodal((500, 500), self.mode)

        if self.logpolar:
            cfg = ect.Config(offset_value_px=self.offset)

            x, y = self.image.shape[0], self.image.shape[1]
            cx, cy = x//2, y//2
            r = min(cx, cy)

            filt = ect.sidelobe(self.dsize, cfg)        
            self.image = ect.logpolar_new(self.image, (cx, cy), self.dsize, r, cfg)
            self.image *= filt
    

    def generate(self) -> np.ndarray:
        return self.image
    

    def images(self) -> Iterable[np.ndarray]:
        for _ in range(self.iterlen):
            yield self.image

@dataclass(slots=True)
class RandUnimodalGenerator(Generator):
    iterlen: int = 100
    offset: int = 10
    dsize: tuple[int, int] = (314, 100)
    image: np.ndarray = field(init=False)
    images: list[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.images = [self._make_image() for _ in range(self.iterlen)]

    def _make_image(self) -> np.ndarray:
        self.image = sine_unimodal((500, 500), self._random_mode())

        cfg = ect.Config(offset_value_px=self.offset)

        x, y = self.image.shape[0], self.image.shape[1]
        cx, cy = x//2, y//2
        r = min(cx, cy)

        filt = ect.sidelobe(self.dsize, cfg)        
        self.image = ect.logpolar_new(self.image, (cx, cy), self.dsize, r, cfg)
        self.image *= filt
        return self.image
    
    def _random_mode(self) -> Mode:
        return Mode(
            period = random.randrange(30, 200, 10),
            angle = random.randrange(75, 105, 5)
        )

    def generate(self) -> np.ndarray:
        self.image = random.choice(self.images)
        return self.image
    

    def images(self) -> Iterable[np.ndarray]:
        for _ in range(self.iterlen):
            self.image = self._make_image()
            return self.image

            