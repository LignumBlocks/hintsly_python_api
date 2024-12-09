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

class Base_Chat:
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
        self.chat_history = [AIMessage(content="Hello")]
        self.system_message = SystemMessage(content="You are a personal assistant of Hintsly, your main task is to help users to navigate this web page")

    def chat_with_history(self, user_input):
        prompt = ChatPromptTemplate.from_messages([self.system_message, MessagesPlaceholder(variable_name="chat_history")])
        self.chat_history.append(HumanMessage(content=user_input))
        formatted_prompt = prompt.format(chat_history=self.chat_history)
        # print(f"formatted_prompt {formatted_prompt}")
        response = self.llm.invoke(formatted_prompt)
        print(f"{response.content}")
        self.chat_history.append(AIMessage(content=response.content))
        return response