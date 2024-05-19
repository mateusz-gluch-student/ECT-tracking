from dataclasses import dataclass, field
import numpy as np

from ..transformers import Transformer

@dataclass(slots=True)
class RelativeCartesianTracer:
    transformer: Transformer
    absolute_position: list[tuple[float, float]] = field(default_factory=list)
    i: int = field(init=False, default=0)

    def __post_init__(self):
        self.absolute_position.append((0,0))

    def callback(self, inp: np.ndarray) -> None:
        '''
        Calculates position of maximum in an 2D array
        and prints output into terminal

        Parameters
        ----------
        image : np.ndarray
            Input Image
        '''

        inv: np.ndarray = self.transformer.invert(inp, logpolar=False, ect=False)
        inv_center: np.ndarray = self._center(inv)

        max_idx = np.argmax(inv_center)
        my, mx = np.unravel_index(max_idx, inv.shape)
        cy, cx = inv.shape[0]//2, inv.shape[1]//2
        
        dy, dx = cy - my, mx - cx
        
        prevx, prevy = self.absolute_position[-1]
        self.absolute_position.append((prevx+dx, prevy+dy))

        self.i += 1
        print(f"Image {self.i}. Delta: {dx = }, {dy = }")



    def _center(self, inp: np.ndarray) -> np.ndarray:
        X, Y = inp.shape
        out = np.zeros_like(inp)
        out[:X//2, :Y//2] = inp[X//2:, Y//2:]
        out[X//2:, :Y//2] = inp[:X//2, Y//2:]
        out[X//2:, Y//2:] = inp[:X//2, :Y//2]
        out[:X//2, Y//2:] = inp[X//2:, :Y//2]
        return out

    @property
    def position(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.absolute_position)
        return pos[:, 0], pos[:, 1]

