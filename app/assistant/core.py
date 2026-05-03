from app.assistant.config import AssistantConfig


class Assistant:
    def __init__(self, config: AssistantConfig):
        self.config = config

    def respond(self, user_input):
        cleaned = user_input.strip()
        if not cleaned:
            return "Digite um comando para a Valeria."

        return (
            f"{self.config.assistant_name} recebeu: {cleaned}. "
            "O nucleo inteligente sera conectado em uma proxima etapa."
        )
