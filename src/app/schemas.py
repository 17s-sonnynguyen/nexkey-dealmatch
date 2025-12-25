from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    top_n: int = Field(50, ge=10, le=500)

class Deal(BaseModel):
    property_id: int
    deal_type: str
    city: str
    state: str
    beds: float
    baths: float
    sqft: float
    purchase_price: float
    arv: float
    entry_fee: float
    estimated_monthly_payment: float
    rerank_score: float
    retrieval_sim: float

class ChatResponse(BaseModel):
    reply: str
    deals: List[Deal] = []
    needs_clarification: bool = False
    missing_fields: Optional[List[str]] = None
