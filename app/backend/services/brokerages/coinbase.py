from app.services.brokerages.base import Brokerage
from app.models.trading import OrderRequest, OrderUnit
import os

class CoinbaseService(Brokerage):
    def execute_order(self, order: OrderRequest):
        params = {
            "product_id": f"{order.ticker}-USD",
            "side": order.side.upper(),
        }
        
        if order.unit == OrderUnit.SHARES:
            params["base_size"] = str(order.amount) # Quantity of BTC/Stock
        else:
            params["quote_size"] = str(order.amount) # Amount of USD
            
        # self.client.create_order(**params)
        return {"status": "success", "engine": "Coinbase-v3-API"}