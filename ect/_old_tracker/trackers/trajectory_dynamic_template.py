from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt

from .tracker import Tracker
from ..loaders import FilepathLoader, Loader
from ..matchers import RotShiftMatcher, Matcher
from ..transformations import UnfilteredTransformation
from ..center import center_of_mass

from dataclasses import dataclass, field

@dataclass
class DynamicTemplateTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    trajectory: list[tuple[float]] = field(default_factory=list)

    def setup(self, **params):
        
        if self.loader is None:
            self.loader = FilepathLoader(
                radius = self.radius,
                offset = self.offset
            )

        if self.matcher is None:
            self.matcher = RotShiftMatcher(
                transformer = UnfilteredTransformation(
                    self.offset,
                    self.ect_offset
                ),
                center_fnc = center_of_mass,
                bp_thresh = 0.1
            )
        
        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)

    def match(self, image: ndarray):
        result = super().match(image)
        scale, angle = self.match_result
        print(f"{scale=}, {angle=}")
        self.trajectory.append((scale, angle))
        return result


    def show_result(self, label: str):
        
        trajectory = np.array(self.trajectory)

        plt.figure(figsize=(10, 10))
        plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.xlabel("Relative scale factor")
        plt.ylabel("Relative rotation")
        plt.title(label)


    def dump_result(self, filepath: str):
        pass
