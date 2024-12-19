from abc import ABC, abstractmethod
from overrides import overrides

import os


class BaseGuardrail(ABC):
    @abstractmethod
    def __call__(self, text: str) -> bool:
        """
        detects if input is harmful/inappropriate.
        """
        return False


class NoGuardrail(BaseGuardrail):

    @overrides
    def __call__(self, text: str) -> bool:
        return True


class SelfIEGuardrail(BaseGuardrail):
    # TODO: Zhongkai will handle this.

    def __init__(self):
        pass

    @overrides
    def __call__(self, text: str) -> str:
        pass


class KeywordFilterGuardrail(BaseGuardrail):

    def __init__(self):
        # Load harmful keyword set from a text file
        with open(os.getcwd() + "/agent_project/verified_harmful_keywords.txt", "r") as f:
            self.harmful_keywords_llm = set(line.strip() for line in f)

    @overrides
    def __call__(self, text: str) -> bool:
        # Return True if any harmful keyword is found in the input text
        return any(phrase.lower() in text.lower() for phrase in self.harmful_keywords_llm)


