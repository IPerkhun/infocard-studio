import os

from dotenv import load_dotenv
from langchain_gigachat import GigaChat

load_dotenv()


class LLMManager:
    def __init__(self):
        self.credentials = os.environ.get("GIGACHAT_CLIENT_SECRET")
        self.scope = os.environ.get("GIGACHAT_SCOPE")

        self.llm = self._init_llm()

    def _init_llm(self) -> GigaChat:
        m = GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            verify_ssl_certs=False,
            model="gigachat-max"
        )
        return m