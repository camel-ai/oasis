from camel.toolkits import FunctionTool

class TransportationAction:

    def __init__(self, agent_id: int, channel: Channel):
        self.agent_id = agent_id
        self.channel = channel

    def get_openai_function_list(self) -> list[FunctionTool]:
        return [
            FunctionTool(func) for func in [
                self.find_route,
                self.get_onto_bus,
                self.get_off_bus,
                self.get_onto_train,
                self.get_off_train,
            ]
        ]
