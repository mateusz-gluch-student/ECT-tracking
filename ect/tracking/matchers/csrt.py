from dataclasses import dataclass, field

from numpy import ndarray

from .matcher import Matcher


@dataclass(slots=True)
class CSRTMatcher(Matcher):

    def initialize(self, input: ndarray):
        return super().initialize()
    
    def match(self, input: ndarray) -> ndarray:
        return super().match()
    
    def update(self, input: ndarray, output: ndarray):
        return super().update(output)
