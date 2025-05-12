from abc import ABCMeta, abstractmethod
from app.modules.decision.interface.schema.decision_schema import InputSchema
class IDecisionRepository(metaclass=ABCMeta):

    @abstractmethod
    def save_decision(self, body: InputSchema, result):
        raise NotImplementedError


