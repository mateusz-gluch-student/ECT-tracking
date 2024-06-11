from dataclasses import dataclass, field
# from pydantic import BaseModel, field

from typing import Callable, Iterable
import numpy as np

from .matchers import Matcher
from .transformers import Transformer
from ..helpers import Generator

@dataclass(slots=True)
class Tracker():
    generator: Generator
    matcher: Matcher
    callback: Callable[[np.ndarray], None] = field(default=lambda x: None)
    outputs: list[np.ndarray] = field(init=False, default_factory=list)


    def track(self) -> Iterable[np.ndarray]:

        init = True

        for image in self.generator.images():

            if init:
                self.matcher.initialize(image)
                init = False
                continue

            yield self.track_single(image)


    def track_single(self, image: np.ndarray):
        match_out = self.matcher.match(np.copy(image))
        self.outputs.append(match_out)
        self.callback(match_out)
        tpl = self.matcher.update(np.copy(image), match_out)
        return image, match_out, tpl
