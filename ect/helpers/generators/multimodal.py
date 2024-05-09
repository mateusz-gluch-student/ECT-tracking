from dataclasses import dataclass, field
from typing import Iterable
import cv2
import numpy as np

import random

import ect
from .generator import Generator
from ...helpers import sine_multimodal, Mode    


@dataclass(slots=True)
class MultimodalGenerator(Generator):
    modes: list[Mode]
    iterlen: int = 100
    offset: int = 10
    dsize: tuple[int, int] = (314, 100)
    image: np.ndarray = field(init=False)


    def __post_init__(self):
        self.image = sine_multimodal((500, 500), self.modes)

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

# @dataclass(slots=True)
# class RandUnimodalGenerator(Generator):
#     iterlen: int = 100
#     offset: int = 10
#     dsize: tuple[int, int] = (314, 100)
#     image: np.ndarray = field(init=False)


#     def _make_image(self) -> np.ndarray:
#         self.image = sine_unimodal((500, 500), self._random_mode())

#         cfg = ect.Config(offset_value_px=self.offset)

#         x, y = self.image.shape[0], self.image.shape[1]
#         cx, cy = x//2, y//2
#         r = min(cx, cy)

#         filt = ect.sidelobe(self.dsize, cfg)        
#         self.image = ect.logpolar_new(self.image, (cx, cy), self.dsize, r, cfg)
#         self.image *= filt
#         return self.image
    
#     def _random_mode(self) -> Mode:
#         return Mode(
#             period = random.randrange(30, 60, 10),
#             angle = random.randrange(75, 105, 5)
#         )

#     def generate(self) -> np.ndarray:
#         return self._make_image()
    

#     def images(self) -> Iterable[np.ndarray]:
#         for _ in range(self.iterlen):
#             yield self._make_image()

            