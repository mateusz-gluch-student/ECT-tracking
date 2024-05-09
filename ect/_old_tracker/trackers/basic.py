from numpy import ndarray
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .tracker import Tracker
from ..matchers import Matcher, BasicMatcher
from ..loaders import Loader, FilepathLoader
from ..transformations import UnfilteredTransformation
from ..center import max_index

@dataclass
class BasicTracker(Tracker):

    template_path: str | None = None

    loader: Loader | None = None
    matcher: Matcher | None = None
    
    radius: float = 200
    offset: float = 10
    ect_offset: float = 10

    result: ndarray | None = None
    ect_template: ndarray | None = None

    def setup(self):

        if self.loader is None:
            self.loader = FilepathLoader(
                filepath = self.template_path,
                radius   = self.radius,
                offset   = self.offset
            )
        
        if self.matcher is None:
            self.matcher = BasicMatcher(
                transformer = UnfilteredTransformation(
                    self.img_offset, 
                    self.ect_offset),
                center_fnc = max_index,
                bp_thresh = 0.1
            )

        self.ect_template = self.loader.load(self.template_path)
        self.ect_template = self.matcher.prepare(self.ect_template)
        

    def show_result(self, label: str, image: ndarray): 
        
        x, y = self.match_result
        print(f"{label}: {x=}, {y=}")

        plt.figure(figsize=(10, 10))
        plt.imshow(self.result)
        plt.title(label)

