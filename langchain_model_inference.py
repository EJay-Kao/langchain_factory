#%%
import os
from typing import List, Union
import pandas as pd
from dotenv import load_dotenv
import json
from tqdm import tqdm

# langchain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from icecream import ic

#%%

# get & check environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)


# 取得環境變數的值
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')


def check_env_var(var_name):
    if var_name in os.environ:
        print(f"{var_name} is set in os.environ: {os.environ[var_name]}")
    else:
        print(f"{var_name} is NOT set in os.environ")

check_env_var('AZURE_OPENAI_API_KEY')
check_env_var('AZURE_OPENAI_ENDPOINT')

#%%

# dictionary, 存放模型資訊
model_api_dc = {
    "gpt-35-turbo": {
        "type": "openai",
        "AZURE_OPENAI_API_VERSION": "***",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "***-gpt-35-turbo-0613"
    },
    "gpt-4": {
        "type": "openai",
        "AZURE_OPENAI_API_VERSION": "***",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "***-gpt-4-0125-preview"
    },
    "gpt-4o": {
        "type": "openai",
        "AZURE_OPENAI_API_VERSION": "***",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": "***-gpt-4o-2024-05-13"
    },
    "ollama-1": {
        "type": "ollama",
        "OLLAMA_API_VERSION": "v1",
        "OLLAMA_CHAT_DEPLOYMENT_NAME": "ollama-1"
    }
}

with open('model_api_dc.json', 'w') as json_file:
    json.dump(model_api_dc, json_file, indent=4)

print("JSON 檔案已成功寫入。")

#%%
class ModelFactory:
    def __init__(self, json_path='model_api_dc.json'):
        self.json_path = json_path
        self.model_info = self._load_model_info()
    
    def _load_model_info(self):
        file_path = os.path.join(os.path.dirname(__file__), self.json_path)
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
            return None
    
    def create_model(self, model_name):
        if not self.model_info or model_name not in self.model_info:
            raise ValueError(f"Model '{model_name}' is not available.")
        
        model_data = self.model_info[model_name]
        
        if model_data["type"] == "openai":
            return AzureChatOpenAI(
                openai_api_version=model_data['AZURE_OPENAI_API_VERSION'],
                azure_deployment=model_data['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']
            )
        elif model_data["type"] == "ollama":
            return OllamaChat(
                api_version=model_data['OLLAMA_API_VERSION'],
                deployment_name=model_data['OLLAMA_CHAT_DEPLOYMENT_NAME']
            )
        else:
            raise ValueError(f"Unknown model type '{model_data['type']}' for model '{model_name}'.")



# %%
class ModelFactory:
    def __init__(self, json_path='model_api_dc.json'):
        self.json_path = json_path
        self.model_info = self._load_model_info()
    
    def _load_model_info(self):
        file_path = os.path.join(os.path.dirname(__file__), self.json_path)
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return None
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON from the file.")
            return None
    
    def create_model(self, model_name):
        if not self.model_info or model_name not in self.model_info:
            raise ValueError(f"Model '{model_name}' is not available.")
        
        model_data = self.model_info[model_name]
        
        if model_data["type"] == "openai":
            return AzureChatOpenAI(
                openai_api_version=model_data['AZURE_OPENAI_API_VERSION'],
                azure_deployment=model_data['AZURE_OPENAI_CHAT_DEPLOYMENT_NAME']
            )
        elif model_data["type"] == "ollama":
            return OllamaChat(
                api_version=model_data['OLLAMA_API_VERSION'],
                deployment_name=model_data['OLLAMA_CHAT_DEPLOYMENT_NAME']
            )
        else:
            raise ValueError(f"Unknown model type '{model_data['type']}' for model '{model_name}'.")

class LanguageModelHandler:
    def __init__(self, model_name, user_question: Union[str, List[str]], system_prompt: Union[str, List[str]] = None, factory=None):
        self.model_name = model_name
        self.user_question = user_question
        self.system_prompt = system_prompt
        self.factory = factory or ModelFactory()
        self._initialize()
    
    def _init_model_api(self):
        """
        initial model API
        """
        self.model = self.factory.create_model(self.model_name)
        # print('Model API initialized.')

    def _init_chain(self):
        """
        initial chain
        """
        default_template = """
        answer back with string format

        system_prompt:
        {system_prompt}

        user_question:
        {user_question}
        """

        parser = StrOutputParser()
        prompt = PromptTemplate(
            template=default_template,
            input_variables=["system_prompt", "user_question"],
        )

        self.chain = prompt | self.model | parser

        # class Data(BaseModel):
        #     input: str = Field(description='user input')
        #     output: List[str] = Field(description='user output')

        # default_template = """
        # 請以json格式回答，
        # key值為content，value為回答內容。
        # 如果system prompt有規定格式，則以system_prompt為準

        # system_prompt:
        # {system_prompt}

        # user_question:
        # {user_question}
        # """

        # parser = JsonOutputParser(pydantic_object=Data)
        # prompt = PromptTemplate(
        #     template=default_template,
        #     input_variables=["system_prompt", "user_question"],
        #     partial_variables={"format_instructions": parser.get_format_instructions()},
        # )

        # self.chain = prompt | self.model | parser
        # print('Chain initialized.')

    def _initialize(self):
        self._init_model_api()
        self._init_chain()

    def get_model_inference(self):
        """
        支持batch or invoke，
        只是如果是batch, 則system_prompt的list長度需與user_question一致
        """
        if isinstance(self.system_prompt, list) and isinstance(self.user_question, list):
            if len(self.system_prompt) != len(self.user_question):
                raise ValueError("The length of system_prompts and user_questions must be the same.")
            self.response = self.chain.batch([
                {"system_prompt": sp, "user_question": uq}
                for sp, uq in zip(self.system_prompt, self.user_question)
            ])
        elif isinstance(self.system_prompt, list) or isinstance(self.user_question, list):
            raise ValueError("Both system_prompt and user_question must be either lists or single strings.")
        else:
            self.response = self.chain.invoke({
                "system_prompt": self.system_prompt,
                "user_question": self.user_question
            })
        return self.response

