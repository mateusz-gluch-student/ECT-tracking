from dataclasses import dataclass, field
from typing import Any, Iterable

@dataclass(slots=True)
class Rect:
    ll: tuple[int, int]
    ul: tuple[int, int]
    lr: tuple[int, int]
    ur: tuple[int, int]

@dataclass(slots=True, frozen=True)
class GroundTruth:
    path: str
    truth: list[Any] = field(init=False, default_factory=list)

    def __post_init__(self):
        with open(self.path) as f:
            for line in f.readlines():
                self.truth.append([float(x) for x in line.split(",")])

    def gettruth_iter(self) -> Iterable[Rect]:
        for t in self.truth:
            yield Rect((int(t[0]), int(t[1])), (int(t[2]), int(t[3])))

    def gettruth(self, i: int) -> Rect:
        t = self.truth[i]
        lr = (int(t[0]), int(t[1]))
        ll = (int(t[2]), int(t[3]))
        ul = (int(t[4]), int(t[5]))
        ur = (int(t[6]), int(t[7]))

        return Rect(ll, ul, lr, ur)
