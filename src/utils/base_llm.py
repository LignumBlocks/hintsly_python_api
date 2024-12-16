import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder  # For defining prompts
from langchain.schema.output_parser import StrOutputParser  # For parsing output
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

MODELS = ['gemini-1.5-flash', 'gemini-1.5-flash-8b']

class Base_LLM:
    """
    A class to manage interactions with different LLMs through langchain framework.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash-8b", temperature: float = 0.7):
        """
        Initializes the Chat with the specified model and temperature.
            
        Args:
            model_name (str): The name of the LLM model to use (default is "gemini-1.5-flash-8b").
            temperature (float): Controls the randomness of model outputs (default is 0.7).
        """
        self.model_name = model_name if model_name else MODELS[0]
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)

    def run(self, user_input, system_prompt=''):
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=system_prompt), HumanMessage(content=user_input)])
        print(prompt)
        response = self.llm.invoke(prompt.messages)
        return response.content


def load_prompt(*args):
    """
    Constructs a prompt by loading the content from one or more prompt template files in the prompts directory.

    Args:
        args (str): The file paths of the prompt templates to load.

    Returns:
        str: The combined content of the loaded prompt templates.
    """

    prompt = ""
    for file_path in args:
        with open(file_path, "r") as file:
            prompt += file.read().strip()
    return prompt