from dependency_injector import containers, providers

from app.database import get_db, SessionLocal
from app.modules.decision.application.decision_service import DecisionService
from app.modules.decision.infra.decision_repo_impl import DecisionRepository

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        packages=[
            "app.modules.decision",  # 의존성을 사용하는 모듈
        ]
    )

    # 의존성 정의
    # db = providers.Resource(get_db)
    decision_repo = providers.Factory(DecisionRepository)
    track_repo = providers.Factory(DecisionRepository)
    decision_service = providers.Factory(
        DecisionService,
        decision_repo=decision_repo
    )

