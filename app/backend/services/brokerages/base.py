from abc import ABC, abstractmethod
from app.models.trading import OrderRequest

class Brokerage(ABC):
    @abstractmethod
    def execute_order(self, order: OrderRequest):
        pass