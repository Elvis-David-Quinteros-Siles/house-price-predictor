from pydantic import BaseModel, Field
from typing import List

class HousePredictionRequest(BaseModel):
    sqft: float = Field(..., gt=0, description="Square footage of the house")
    bedrooms: int = Field(..., ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(..., gt=0, description="Number of bathrooms")
    location: str = Field(..., description="King County zipcode (e.g. 98103); see /model-info")
    year_built: int = Field(..., ge=1900, le=2015, description="Year the house was built")
    condition: str = Field(..., description="Poor, Fair, Average, Good or Very Good")

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    features_importance: dict
    prediction_time: str
    latency_ms: float