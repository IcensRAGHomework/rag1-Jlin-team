import json
import traceback
import requests
import base64
import re
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import Dict, List
from mimetypes import guess_type
from openai import AzureOpenAI

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)


llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )


def generate_hw01(question: str):
    #Define a prompt template

    prompt_template = """
    你是一個專門生成數據的助手，請按照以下格式輸出資料，並以繁體中文呈現,如果有多筆資料也需輸出：
    Do not include any additional text like "json:" or comments.
        
    {{
        "Result": [
            {{
                "date": "2024-01-01",
                "name": "元旦"
            }},
            {{
                "date": "2024-01-01",
                "name": "元旦"
            }}
            ]
    }}
    
    請生成{question}的資料：
    """
    
    prompt = prompt_template.format(question=question)
    response = llm.invoke(prompt)
    return response.content

memory_store = {}
    
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
        """In memory implementation of chat message history."""
        
        messages: List[BaseMessage] = Field(default_factory=list)

        def add_messages(self, messages: List[BaseMessage]) -> None:
             """Add a list of messages to the store"""
             self.messages.extend(messages)
        
        def clear(self) -> None:
             self.messages = []
    
def get_by_session_id(session_id: str) -> InMemoryHistory:
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryHistory()
        return memory_store[session_id]
    
history = get_by_session_id("1")

conversation_contents = [
    {"input":"今年台灣1月紀念日有哪些?", 
     "output":"""{"Result": [{"date": "2024-01-01","name": "元旦"}]}"""},
    {"input":"""根據先前的節日清單，這個節日{"date": "01-01", "name": "元旦"}是否有在該月份清單？""", 
     "output":"""{"Result":{"add": false, "reason": "元旦已包含在一月的節日清單中"}"""},
    {"input":"今年台灣10月紀念日有哪些?", 
     "output":"""{"Result": [{"date": "2024-10-10","name": "國慶日"}]}"""},
    {"input":"""根據先前的節日清單，這個節日{"date": "10-31", "name": "蔣公誕辰紀念日"}是否有在該月份清單？""", 
     "output":"""{"Result":{"add": true, "reason": "蔣中正誕辰紀念日並未包含在十月的節日清單中。目前十月的現有節日包括國慶日、重陽節、華僑節、台灣光復節和萬聖節。因此，如果該日被認定為節日，應該將其新增至清單中。"}"""}
]
    
history_cases = []
for contents in conversation_contents:
         history_cases.append(HumanMessage(contents["input"]))
         history_cases.append(AIMessage(contents["output"]))
history.add_messages(history_cases)

def image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    
    base64_data = base64.b64encode(image_data).decode('utf-8')
    
    data_url = f"data:{mime_type};base64,{base64_data}"
    return data_url

def generate_hw02(question):
    # define a prompt template for parsing the question
    parse_prompt = PromptTemplate(
        input_variables=["question"],
        template = """
         You are an assistant that extracts relevant details from a question to query an API.
         Do not include any additional text like "json:" or comments.
         The user will ask a question about holidays. Extract the following details:
         - Year (e.g., 2024)
         - Month (e.g., October)
         - Country (ISO 3166-1 alpha-2 code, e.g., 'US' for the United States or 'TW' for Taiwan)

        Question: "{question}"
        
        Provide the extracted details in JSON format:
        {{
            "year": "string",
            "month": "number",
            "country": "string"
        }}
        """
        )
    
    def parse_question(question: str):
            prompt = parse_prompt.format(question=question)
            response=llm.invoke(prompt)
            return response.content
        
    def get_params_from_question(question: str):
            params_details = parse_question(question)
            parsed_detail_json = json.loads(params_details)
            return parsed_detail_json
            
    API_URL = "https://calendarific.com/api/v2/holidays"
    API_KEY = "PoXlCCF8EfYOO5cVDOE0rIF4ZcTZbnlM"
    def use_params_to_call_api(question: str):
            parsed_data = get_params_from_question(question)
            year = parsed_data["year"]
            month = parsed_data["month"]
            country = parsed_data["country"].lower()
            
            params={"api_key": API_KEY, "year":year, "month": month, "country": country}
            
            
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            return data["response"]["holidays"]
           
    
    parse_result_prompt_template = """
    你是一個專門生成數據的助手，請根據{api_result}取得節日日期與名稱,並按照以下格式輸出資料，並翻譯成繁體中文呈現,如果有多筆資料也需輸出：
    Do not include any additional text like "json:" or comments.
        
    {{
        "Result": [
            {{
                "date": "2024-01-01",
                "name": "元旦"
            }},
            {{
                "date": "2024-01-01",
                "name": "元旦"
            }}
            ]
    }}
    
    請生成{api_result}符合格式的資料：
    """
    def extract_api_result_data_by_question(question: str):
        api_result = use_params_to_call_api(question)
        prompt = parse_result_prompt_template.format(api_result=api_result)
        response = llm.invoke(prompt)
        return response.content
    
    return extract_api_result_data_by_question(question)

def generate_hw03(question2, question3):
    history.add_user_message(question2)
    history.add_ai_message(generate_hw02(question2))
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You know all holidays, and you will only return valid JSON without enclosing it in '''json'''"),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "{input}")
        ]
        )
    
    chat_history = RunnableWithMessageHistory(
        prompt | llm,
        get_by_session_id,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    response = chat_history.invoke(
        {"input": question3},
        config={"configurable": {"session_id": "1"}}
    )
    
    content = json.loads(response.content)
    return json.dumps(
                {
                    "Result": {
                        "add": content['Result']['add'],
                        "reason": content['Result']['reason'],
                    }
                }
            )

def generate_hw04(question):
    image_path = image_to_data_url("baseball.png")
    client = AzureOpenAI(
        api_key=gpt_config['api_key'],
        api_version=gpt_config['api_version'],
        base_url=f"{gpt_config['api_base']}openai/deployments/{gpt_config['deployment_name']}"
    )
    
    response = client.chat.completions.create(
        model = gpt_config['deployment_name'],
        messages=[
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": question
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": image_path
                }
            }
        ] } 
    ],
    max_tokens=2000 
    )
    
    content = re.search(r"(\d+)", response.choices[0].message.content)
    score = int(content.group(1))
    return json.dumps({"Result":{"score":score}})
    
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    print(response)