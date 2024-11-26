from decouple import config
from langchain_openai import ChatOpenAI

from core.helpers.model import is_reasoning_model_enabled
from core.vendor.base import BaseLLM
from langchain.agents.openai_assistant import OpenAIAssistantRunnable


class OpenAI(BaseLLM):
    _instance = None
    _assistant_name = "Inteliome Assistant"
    _assistant_instruction = "You are an intelligent agent to help users on retrieval and analysis of structured and unstructured data."

    def __init__(
        self,
        openai_api_key: str = None,
        name: str = None,
        instruction: str = None,
        model: str = None,
        temperature: float = None,
        stream: bool = False,
    ):
        """
        Initializes a new instance of the OpenAI connection class.

        Args:
            name (str): The name of assistant to use.
            instruction (str): The instruction to the assistant.
            model (str): The OpenAI model to use. Defaults to the value of the 'OPENAI_MODEL' environment variable.
            temperature (float): The temperature value for the OpenAI connection. Defaults to 0.3.
            openai_api_key (str): The OpenAI API key. Defaults to the value of the 'OPENAI_API_KEY' environment variable.
            stream (bool): Whether to stream the OpenAI connection. Defaults to False.

        Returns:
            None
        """
        self._reasoning_llm = None
        self._name = name
        self._instruction = instruction
        self._openai_api_key = openai_api_key if openai_api_key else config("API_KEY")
        self._model = model if model else config("MODEL")
        self._temperature = temperature if temperature else config('TEMPERATURE')
        self._stream = stream
        self._llm=ChatOpenAI(api_key=self._openai_api_key, model=self._model)
        self._create_llm()
        self._assistant = None

        if is_reasoning_model_enabled():
            if config('REASONING_MODEL') is None:
                raise(ValueError('Reasoning model not set. Please set REASONING_MODEL in env variable.'))

            self._reasoning_llm = config('REASONING_MODEL')
            self._create_reasoning_llm()

    def __new__(cls, *args, **kwargs):
        """
        Creates a new instance of the class if one does not already exist, otherwise returns the existing instance.

        Args:
            cls (type): The class object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            object: The instance of the class.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def _create_llm(self, model = None, temperature = None):
        model = model if model is not None else self._model
        temperature = temperature if temperature is not None else self._temperature
        if model != 'o1-preview' and model != 'o1-mini':
            llm = ChatOpenAI(openai_api_key=self._openai_api_key, temperature=temperature, model_name=model)
        else:
            llm = ChatOpenAI(
                api_key = self._openai_api_key, model_name=model
            )
        self._llm = llm

    def _create_reasoning_llm(self, model = None):
        model = model if model is not None else self._reasoning_llm
        reasoning_llm = ChatOpenAI(openai_api_key=self._openai_api_key, model_name=model)
        self._reasoning_llm = reasoning_llm

    def __create_assistant__(self):
        self._assistant = OpenAIAssistantRunnable.create_assistant(
            name= self._name if self._name else OpenAI._assistant_name,
            instructions= self._instruction if self._instruction else OpenAI._assistant_instruction,
            model= self._model,
            temperature = self._temperature,
            tools = []
        )

    def get_llm(self):
        """
        Retrieves a language model (LLM) instance from the connection pool.

        Returns:
            LLM: The language model instance.
        """
        if not self._llm:
            raise ValueError("LLM has not been created.")

        return self._llm

    def get_reasoning_llm(self):
        """
        Retrieves a language model (LLM) instance from the connection pool.

        Returns:
            LLM: The language model instance.
        """
        if not self._reasoning_llm:
            raise ValueError("Reasoning LLM has not been created.")

        return self._reasoning_llm

    def get_assistant(self):
        """
        Retrieves a language model (LLM) instance from the connection pool.

        Returns:
            LLM: The language model instance.
        """
        if not self._assistant:
            raise ValueError("Assistant has not been created.")

        return self._assistant

    def completion(self, prompt: str, mode = 'llm'):
        """
        Generates a completion for a given prompt using the specified model and temperature.

        Args:
            prompt (str): The input prompt to be completed.

        Returns:
            str: The generated completion.

        Raises:
            Exception: If an error occurs during the completion process.
        """
        try:
            if mode == 'llm':
                result = self._llm.invoke(prompt)
                return result.content

            result = self._assistant.invoke(prompt)
            return result.content

        except Exception as e:
            raise f"LLM is unable to process your query: {e}"
