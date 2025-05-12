from enum import Enum
from fastapi import HTTPException, status


class ErrorCode(Enum):
    # Decision 관련 에러 코드
    DECISION_CREATED_FAILED = (status.HTTP_409_CONFLICT, "Decision 생성 실패.")
    DECISION_FAILED = (status.HTTP_409_CONFLICT, "Decision 실패.")
    DECISION_TRANSFER_FAILED = (status.HTTP_409_CONFLICT, "Decision Inputschema 변형 실패.")

    def __init__(self, code, message):
        self.code = code
        self.message = message
