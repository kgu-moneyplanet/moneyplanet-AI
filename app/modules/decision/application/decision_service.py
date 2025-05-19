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
        ai_input_data = self.transfer_to_ai_input_data(body)
        ai_output_data = models_crud.predict_from_input(ai_input_data)
        #self.create_decision(body, ai_output_data)
        return ai_output_data

    def gender_parser(self, gender: str):
        if gender == "M":
            return "남성"
        if gender == "F":
            return "여성"
        raise raise_error(ErrorCode.DECISION_TRANSFER_FAILED)

    def planet_parser(self, planet: str):
        if planet == "MERCURY":
            return "수성"
        if planet == "EARTH":
            return "지구"
        if planet == "MARS":
            return "화성"
        if planet == "JUPITER":
            return "목성"
        if planet == "SATURN":
            return "토성"
        if planet == "URANUS":
            return "천왕성"
        if planet == "NEPTUNE":
            return "해왕성"
        if planet == "VENUS":
            return "금성"
        raise raise_error(ErrorCode.DECISION_TRANSFER_FAILED)

    def transfer_to_ai_input_data(self, body: InputSchema):
        if body is None:
            raise raise_error(ErrorCode.DECISION_TRANSFER_FAILED)
        planet = self.planet_parser(body.planet)
        gender = self.gender_parser(body.gender)
        spendingdetail = SpendingDetails(
            date = body.tx_date,
            amount = body.amount,
            category = body.category_name,
            description = body.content,
            spending_reason = body.memo)
        userprofile = UserProfile(
            planet = planet,
            gender = gender,
            age = body.age,
            job = body.job,
            user_survey = body.prefer)
        ai_input_data = AiInputData(
            spending_details = spendingdetail,
            user_profile = userprofile)
        return ai_input_data

    def create_decision(self, body: InputSchema, result):
        self.decision_repo.save_decision(body,result)