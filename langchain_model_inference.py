#%%
import os
from typing import List, Union
import pandas as pd
from dotenv import load_dotenv
import json
from tqdm import tqdm

# langchain
from langchain_openai import AzureChatOpenAI  # azure
from langchain.llms import Ollama  # ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

import pandas as pd
from datetime import datetime
from tqdm import tqdm
from icecream import ic



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
            return Ollama(
                model=model_name,
                base_url=model_data['OLLAMA_base_url']
            )
        else:
            raise ValueError(f"Unknown model type '{model_data['type']}' for model '{model_name}'.")


# %%

class LanguageModelHandler:
    def __init__(self, model_name, user_question: Union[str, List[str]], system_prompt: Union[str, List[str]] = None, factory=ModelFactory):
        self.model_name = model_name
        self.user_question = user_question
        self.system_prompt = system_prompt
        self.factory = factory()  # or ModelFactory()
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
        You are a chatbot.
        Your responsibility is to base on system_prompt (if exists) answer user_questions.

        Here's the rule:
        1. answer back with string format
        2. answer should follow question language
        3. do not answer with emoji

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

# %% 

def generate_inference_output(input_csv, model_name, test_plan, output_csv):
    """
    output only for error handlng
    """
    # 讀取CSV文件
    df = pd.read_csv(input_csv)

    # 定義列名
    columns = [
        'test_plan', 
        'model_name', 
        'start_time', 
        'end_time',
        'q_set_name',
        'q_id',
        'domain',
        'sys_prompt',
        'user_question',
        'ground_truth',
        'llm_ans',
        'Response_id'
    ]

    # 檢查DataFrame中是否包含所有必需的列
    required_columns = [
        'q_set_name',
        'q_id',
        'domain',
        'sys_prompt',
        'user_question',
        'ground_truth'
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # 創建空的DataFrame
    output = pd.DataFrame(columns=columns)

    # 使用tqdm顯示進度條
    error_row = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            model = LanguageModelHandler(
                model_name=model_name,
                user_question=row['user_question'],
                system_prompt=row['sys_prompt']
            )
            
            LLM_ans = model.get_model_inference()
            end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 構建新的一行數據
            new_row = [
                test_plan, model_name, start_time, end_time,
                row['q_set_name'], 
                row['q_id'], 
                row['domain'],
                row['sys_prompt'],
                row['user_question'], 
                row['ground_truth'], 
                LLM_ans, 
                i
            ]

            # 將新行添加到DataFrame中
            output = pd.concat([output, pd.DataFrame([new_row], columns=columns)], ignore_index=True)
        except:
            error_row.append(i)

            

    # 將結果保存到CSV文件
    output.to_csv(output_csv, index=False, encoding='utf-8')
    print('Done')
    return error_row