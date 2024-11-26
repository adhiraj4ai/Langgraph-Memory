from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def get_llm(self):
        """
        Abstract method to retrieve a Large Language Model (LLM).

        Returns:
            LLM: A Large Language Model instance.
        """
        NotImplementedError

    @abstractmethod
    def completion(self, prompt: str):
        """
        Abstract method to generate a completion for a given prompt.

        Args:
            prompt (str): The input prompt to be completed.

        Returns:
            The completion of the given prompt.
        """
        NotImplementedError

    @abstractmethod
    def get_assistant(self):
        """
        Abstract method to retrieve an assistant.

        Returns:
            assistant: An LLM based assistant
        """
        NotImplementedError


class LLM:
    def __init__(self, llm):
        """
        Initializes an instance of the LLM class.

        Args:
            llm: The language model to be used.

        Returns:
            None
        """
        self.llm = llm

    def completion(self, prompt: str):
        """
        This function generates a completion for a given prompt using the LLM model.

        Args:
            prompt (str): The input prompt to be completed.

        Returns:
            str: The completed prompt.
        """
        result = self.llm.completion(prompt)
        return result
