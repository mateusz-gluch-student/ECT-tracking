from pydantic import BaseModel, Field
from dataclasses import dataclass

import math
from typing import Literal, Any

DEFAULT_SNF = [1.747, 1.743, 1.731, 1.724, 1.708, 1.710, 
               1.728, 1.723, 1.752, 1.749, 1.762, 1.862, 
               1.888, 1.894, 1.850, 1.861, 1.720, 1.650, 1.578, 1.202]


DEFAULT_FNF = [1.000, 0.000, 1.663, 0.794, 1.161, 0.994, 
               1.246, 1.046, 1.048, 1.018, 1.043, 1.049, 
               1.013, 1.059, 1.012, 1.024, 0.997, 0.871, 0.888, 0.582]


class AntialiasParameters(BaseModel): 
    factor: float
    slope: float = 0.25
    threshold: float = 1
    vector: Any = Field(default=None)

class Config(BaseModel):
    mode: Literal["offset", "omit", "opencv"] = Field(default="offset")
    interpolation: Literal["bilinear", "none"] = Field(default="bilinear")
    start_angle_deg: float = Field(default=90)
    offset_value_px: int = Field(default=10)
    ect_offset_value_px: int = Field(default=1)

    antialias: bool = True
    antialias_params: list[AntialiasParameters] = Field(default = None)
    antialias_factors: tuple[float, float] = [0.398, 0.113]
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
