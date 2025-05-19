from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide
from app.containers import Container
from app.modules.decision.application.decision_service import DecisionService
from app.modules.decision.interface.schema.decision_schema import InputSchema, OutputResponse


router = APIRouter(prefix="/v1/decision", tags=["decision"])

@router.post("/abc", response_model=OutputResponse)
@inject
def decide_abc(
    body: InputSchema,
    decision_service: DecisionService = Depends(Provide[Container.decision_service])
):
    return decision_service.decide_abc(body)

