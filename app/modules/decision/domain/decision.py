from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional

@dataclass
class Decision:
    id: str
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
    memo:str
    abc: str
    reason: str
    feedback: str
    create_datetime: datetime
    update_datetime: datetime