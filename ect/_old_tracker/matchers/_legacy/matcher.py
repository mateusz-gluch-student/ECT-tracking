import cv2

from numpy import ndarray

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..transformations import Transformation
from typing import Callable


@dataclass
class Matcher(ABC):

    transformer: Transformation
    center_fnc: Callable[[ndarray], tuple[int, int]]

    @abstractmethod
    def prepare(
        self,
        image: ndarray
    ) -> ndarray: ...

    @abstractmethod
    def match_image(
        self, 
        image: ndarray, 
        template: ndarray
        ) -> ndarray: ...


    def match_numeric(
            self, 
            image: ndarray,
            template: ndarray
            ) -> tuple[float, float]:

        if self.match_result is None:
            match_result = self.match_image(image, template)
        else:
            match_result = self.match_result

        # print(self.center_fnc.__name__)
        return self.center_fnc(match_result)


    def match_point(self, image:ndarray, template:ndarray) -> ndarray:

        x, y = self.match_numeric(image, template)

        return cv2.circle(
            img = self.match_result,
            center = (int(x), int(y)),
            radius = 10,
            color = (255, 0, 0),
            thickness = 3  
        )
    
    def clear(self):
        self.match_result = None
    