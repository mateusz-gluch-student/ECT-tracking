from dataclasses import dataclass, field
import cv2 
import numpy as np
import os
import random
from typing import Iterable

import ect
from .generator import Generator


@dataclass(slots=True)
class IdSequenceImageGenerator(Generator):
    pathspec: str
    '''dirpath spec with "{id}" format spec'''
    seqlen: int
    idx: int = field(init = False, default=0) 
    _images: list[np.ndarray] = field(init=False, default_factory=list)


    def __post__init__(self):
        for root, _, files in os.walk(self.dirpath):
            for file in files:
                self._load_image(os.path.join(root, file))


    # def _load_image(self, imgpath: str) -> np.ndarray:
    #     img = cv2.imread(imgpath)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
 
    #     x, y = img.shape[0], img.shape[1]
    #     cx, cy = x//2, y//2
    #     r = min(cx, cy)
    #     dsize = (((np.pi*r)//2)*2, r)

    #     filt = ect.sidelobe(
    #         dsize, offset=10, 
    #         flags=ect.ECT_OFFSET_ORIGIN | ect.ECT_GRAYSCALE | ect.ECT_START_NY,
    #         )[:, :, 0]
        
    #     img = ect.logpolar_new(img, (cx, cy), dsize, r)
    #     img *= filt
    #     return img

    def _load_image(self, imgpath: str) -> np.ndarray:
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255
        self._images.append(img)
        return img

    def generate(self) -> np.ndarray:
        for i in range(self.seqlen):
            self._load_image(self.pathspec.format(id=i))

        if self.idx >=len(self.images):
            self.idx = 0

        ret = self.images[self.idx]
        self.idx += 1
        return ret


    def images(self) -> Iterable[np.ndarray]:

        for i in range(self.seqlen):
            yield self._load_image(self.pathspec.format(id=i))
