from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Mode:
    period: float
    angle: float
    amplitude: float = 1

    def __repr__(self) -> str:
        return f"Mode(T={self.period:.3f}, angle={self.angle:.3f}, A={self.amplitude:.3f})"

    @property
    def phi(self) -> float:
        return self.angle*np.pi/180


def sine_unimodal(
    dsize: tuple[int, int], 
    mode: Mode
) -> np.ndarray:
    x = np.linspace(0, dsize[1], dsize[1], endpoint=False)
    y = np.linspace(0, dsize[0], dsize[0], endpoint=False)
    xx, yy = np.meshgrid(x, y)
    return 1 + np.sin(2*np.pi/mode.period*(xx*np.cos(mode.phi)+yy*np.sin(mode.phi)))

def sine_multimodal(
    dsize: tuple[int, int],
    modes: list[Mode]
) -> np.ndarray :
    x = np.linspace(0, dsize[1], dsize[1], endpoint=False)
    y = np.linspace(0, dsize[0], dsize[0], endpoint=False)
    xx, yy = np.meshgrid(x, y)

    out = np.zeros(dsize)
    for m in modes:
        phase = xx*np.cos(m.phi) + yy*np.sin(m.phi)
        out += m.amplitude * np.sin(2*np.pi/m.period*phase)

    return cv2.normalize(out, None, 0, 1, cv2.NORM_MINMAX)


