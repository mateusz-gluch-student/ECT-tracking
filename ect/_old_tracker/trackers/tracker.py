from numpy import ndarray
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..matchers import Matcher
from ..loaders import Loader

@dataclass
class Tracker(ABC):

    loader: Loader
    matcher: Matcher

    def match(self, image: ndarray): 

        self.matcher.clear()

        ect_image = self.matcher.prepare(image)
        self.result = self.matcher.match_point(ect_image, self.ect_template)
        self.match_result = self.matcher.match_numeric(ect_image, self.ect_template)

        return self.result


    def match_file(self, path: str): 
        
        image = self.loader.load(path)
        return self.match(image)
    
    @abstractmethod
    def setup(self, **params): ...


    @abstractmethod
    def show_result(self, label: ndarray, *args, **kwargs): ...