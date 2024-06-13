from pydantic import BaseModel, Field
from dataclasses import dataclass

import math
from typing import Literal, Any

DEFAULT_SNF = [1.6175, 1.7390, 1.9404, 2.1780, 2.2013, 2.2393, 
               2.2169, 2.2180, 2.1818, 2.0851, 2.1757, 2.1821, 
               2.2178, 2.1828, 2.2628, 2.1157, 2.3304, 2.0736, 2.2568, 1.1056]

DEFAULT_FNF = [1.3347, -0.1276, 0.7737, 0.4291, 0.8655, 0.9544, 
               0.9387, 0.9636, 0.9533, 0.9668, 0.9320, 0.9144, 
               0.9395, 0.9503, 0.9899, 0.9316, 0.9708, 0.8407, 1.0149, 0.2355]


class AntialiasParameters(BaseModel): 
    factor: float
    slope: float = 0.25
    threshold: float = 1
    vector: Any = Field(default=None)

class Config(BaseModel):
    mode: Literal["offset", "omit", "opencv"] = Field(default="offset")
    interpolation: Literal["bilinear", "none"] = Field(default="bilinear")
    start_angle_deg: float = Field(default=90)
    offset_value_px: float = Field(default=10)
    ect_offset_value_px: float = Field(default=1)

    antialias: bool = True
    antialias_params: list[AntialiasParameters] = Field(default = None)
    antialias_factors: tuple[float, float] = [0.49, 0.14]
    transform: Literal["ect", "iect"] = Field(default="ect")

    sidelobe_slope: float = 0.25
    freqnorm_knots: list[float] = Field(default=DEFAULT_FNF, frozen=True)
    spacenorm_knots: list[float] = Field(default=DEFAULT_SNF, frozen=True)

    @property
    def start_angle_rad(self) -> float:
        return math.pi * self.start_angle_deg / 180    
    
    @property
    def offset(self) -> int:
        if self.mode == "offset":
            return self.offset_value_px
        elif self.mode == "opencv":
            return self.offset_value_px
        else:
            return 0

    def validate(self):

        # if self.antialias is not None and self.antialias_params is None:
        #     raise AttributeError("antialias needs antialias parameters specified")
        
        if self.mode == "offset" and self.offset_value_px is None:
            raise AttributeError("logpolar offset mode requires an offset value")
        
        if self.mode == "offset" and self.ect_offset_value_px is None:
            raise AttributeError("logpolar offset mode requires a transform offset value")
