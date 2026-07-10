from pydantic import BaseModel


class BugRequest(BaseModel):
    summary: str
    description: str
    component_name: str
    product_name: str
    quantity_of_votes: int = 0
    quantity_of_comments: int = 0


class PredictionResponse(BaseModel):
    severity: str
    estimated_fix_time: float