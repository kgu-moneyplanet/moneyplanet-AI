from dependency_injector.wiring import inject
from app.modules.decision.domain.repository.decision_repo import IDecisionRepository
from app.utils.exceptions.error_code import ErrorCode
from app.utils.exceptions.handlers import raise_error
from app.modules.decision.interface.schema.decision_schema import InputSchema, SpendingDetails, UserProfile, \
    OutputResponse, AiInputData
from app.models.crud import models_crud

class DecisionService:
    @inject
    def __init__(
            self,
            decision_repo: IDecisionRepository
    ):
        self.decision_repo =decision_repo

    def decide_abc(self, body: InputSchema):
        try:
            ai_input_data = self.transfer_to_ai_input_data(body)
            ai_output_data = models_crud.predict_from_input(ai_input_data)
            return ai_output_data
        except Exception as e:
            import traceback
            print("❌ 예외 발생:", e)
            traceback.print_exc()
            raise raise_error(ErrorCode.DECISION_FAILED)

    def transfer_to_ai_input_data(self, body: InputSchema):
        if body is None:
            raise raise_error(ErrorCode.DECISION_TRANSFER_FAILED)
        spendingdetail = SpendingDetails(
            date = body.tx_date,
            amount = body.amount,
            category = body.category_name,
            description = body.content,
            spending_reason = body.memo)
        userprofile = UserProfile(
            planet = body.planet,
            gender = body.gender,
            age = body.age,
            job = body.job,
            user_survey = body.prefer)
        ai_input_data = AiInputData(
            spending_details = spendingdetail,
            user_profile = userprofile)
        return ai_input_data

    def create_decision(self, body: InputSchema, result):
        self.decision_repo.save_decision(body,result)