import os

from dotenv import load_dotenv
from langchain_gigachat import GigaChat

load_dotenv()


class LLMManager:
    def __init__(self):
        self.credentials = os.environ.get("GIGACHAT_CREDENTIALS")
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.verify_ssl = (
            os.environ.get("GIGACHAT_VERIFY_SSL", "false").lower() == "true"
        )

        self.temperature = float(os.environ.get("GIGACHAT_TEMPERATURE", "0.0"))
        self.top_p = float(os.environ.get("GIGACHAT_TOP_P", "1.0"))

        self.llm = self._init_llm()

    def _init_llm(self) -> GigaChat:
        m = GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            verify_ssl_certs=self.verify_ssl,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return m