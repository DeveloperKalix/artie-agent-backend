class FidelityService(Brokerage):
    def execute_order(self, order: OrderRequest):
        # We construct a natural language goal for the TinyFish Agent
        goal = f"{order.side} {order.amount} {order.unit.value} of {order.ticker} on Fidelity."
        
        # response = tinyfish.run_agent(url="https://fidelity.com", goal=goal)
        return {"status": "success", "engine": "TinyFish-Web-Agent", "goal": goal}