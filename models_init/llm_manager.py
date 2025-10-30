import logging

import httpx
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_openai import ChatOpenAI

from const import PARAMETRS_LLM

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self):
        self.base_url = PARAMETRS_LLM["base_url"]
        self.model_name = PARAMETRS_LLM["model_name"]
        self.llm = self._init_llm()

    def _init_llm(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            model=self.model_name,
            api_key="EMPTY",
            base_url=self.base_url,
            temperature=PARAMETRS_LLM['temperature'],
            top_p=PARAMETRS_LLM["top_p"],
            streaming=False,
            http_client=httpx.Client(),
            # callbacks=[ConsoleCallbackHandler()],
        )

        return llm