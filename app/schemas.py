from pydantic import BaseModel

class Probs(BaseModel):
    benign: float
    malignant: float

class InferResponse(BaseModel):
    probs: Probs
    camPngUrl: str