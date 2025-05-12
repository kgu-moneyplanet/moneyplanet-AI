from datetime import datetime, date
from typing import Optional, List

from pydantic import BaseModel, Field


class InputSchema(BaseModel):
    user_id: str
    planet: str
    gender: str
    prefer: Optional[str]
    age: int
    job: str
    tx_date: date
    amount: int
    category_name: str
    content: str
    memo: str

class OutputResponse(BaseModel):
    abc: str
    reason: str
    feedback: str

class SpendingDetails(BaseModel):
    date: date
    amount: int
    category: str
    description: str
    spending_reason: Optional[str]

class UserProfile(BaseModel):
    planet: str
    gender: str
    age: int
    job: str
    user_survey: Optional[str]


class AiInputData(BaseModel):
    spending_details: SpendingDetails
    user_profile: UserProfile