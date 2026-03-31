from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    sex: Literal["F", "M"]
    age: int = Field(..., ge=10, le=30)
    address: Literal["R", "U"]
    famsize: Literal["LE3", "GT3"]
    Pstatus: Literal["A", "T"]

    Medu: int = Field(..., ge=0, le=4)
    Fedu: int = Field(..., ge=0, le=4)

    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]

    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=4)

    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]

    famrel: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0, le=100)

    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)
    G3: int = Field(..., ge=0, le=20)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_label: str
