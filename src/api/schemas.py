from pydantic import BaseModel, Field
from typing import List

class HousePredictionRequest(BaseModel):
    sqft: float = Field(..., gt=0, description="Living area in square feet")
    sqft_lot: float = Field(..., gt=0, description="Lot area in square feet")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., gt=0, description="Number of bathrooms")
    year_built: int = Field(..., ge=1900, le=2015, description="Year the house was built")
    condition: str = Field(..., description="Poor, Fair, Average, Good or Very Good")
    grade: int = Field(..., ge=1, le=13, description="King County construction grade (1-13, 7 = average)")
    view: int = Field(0, ge=0, le=4, description="Quality of the view (0 = none, 4 = excellent)")
    waterfront: bool = Field(False, description="Whether the house is on the waterfront")
    lat: float = Field(..., ge=47.1, le=47.8, description="Latitude (King County, WA)")
    long: float = Field(..., ge=-122.6, le=-121.3, description="Longitude (King County, WA)")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    features_importance: dict
    prediction_time: str
    latency_ms: float