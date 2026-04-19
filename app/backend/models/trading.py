from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class OrderUnit(str, Enum):
    SHARES = "shares"
    DOLLARS = "dollars"

class OrderRequest(BaseModel):
    ticker: str
    side: str  # "buy" or "sell"
    amount: float
    unit: OrderUnit = OrderUnit.DOLLARS