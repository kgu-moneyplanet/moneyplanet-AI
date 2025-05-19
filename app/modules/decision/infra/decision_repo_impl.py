from abc import ABC
from app.database import SessionLocal
from datetime import datetime
from app.modules.decision.domain.repository.decision_repo import IDecisionRepository
from app.modules.decision.domain.decision import Decision as DecisionVO
from app.modules.decision.infra.db_models.decision import Decision
from app.modules.decision.interface.schema.decision_schema import InputSchema
from app.utils.db_utils import row_to_dict


class DecisionRepository(IDecisionRepository, ABC):

    def save_decision(self, body: InputSchema, result):
        with SessionLocal() as db:
            new_decision = Decision(
                user_id = body.user_id,
                planet = body.planet,
                gender = body.gender,
                prefer = body.prefer,
                age = body.age,
                job = body.job,
                tx_date = body.tx_date,
                amount = body.amount,
                category_name = body.category_name,
                content= body.content,
                memo = body.memo,
                abc = result.abc,
                reason = result.reason,
                feedback = result.feedback,
                create_datetime = datetime.utcnow(),
                update_datetime = datetime.utcnow()
            )
            db.add(new_decision)
            db.commit()
            return
