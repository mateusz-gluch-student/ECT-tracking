from dataclasses import dataclass, field
# from pydantic import BaseModel, field

from typing import Callable
import numpy as np

from .matchers import Matcher
from .transformers import Transformer
from ..helpers import Generator

@dataclass(slots=True)
class Tracker():
    generator: Generator
    matcher: Matcher
    transformer: Transformer
    callback: Callable[[np.ndarray], None]
    outputs: list[np.ndarray] = field(init=False, default_factory=list)


    def track(self):

        init = True

        for image in self.generator.images():

            if init:
                tr_image = self.transformer.transform(image)
                self.matcher.initialize(tr_image)
                init = False
                continue

            self.track_single(image)


    def track_single(self, image: np.ndarray):

        tr_image = self.transformer.transform(image)
        match_out = self.matcher.match(tr_image)
        self.outputs.append(match_out)
        self.matcher.update(tr_image, match_out)
        self.callback(match_out)
        return 
