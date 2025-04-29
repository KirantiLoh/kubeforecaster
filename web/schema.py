from pydantic import BaseModel, Field

class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""
    status: str = "OK"
    
class PredictionBody(BaseModel):
    cpu_loads: list[float] = Field(min_length=15)
    memory_loads: list[float] = Field(min_length=15)
   