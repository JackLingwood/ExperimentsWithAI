# https://python.langchain.com/docs/introduction/

from dotenv import load_dotenv
import os
import sys

# Add Shared folder to sys.path
sys.path.append(os.path.abspath("Shared"))

from utils import heading, heading2, heading3, clearConsole
from openai import OpenAI

# Load environment variables
load_dotenv()

# Clear console and set working directory
clearConsole()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

# Headings
name = os.path.basename(__file__)
heading(f"{name} Chat Prompt Templates & Chat Prompt Values")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import (SystemMessagePromptTemplate,
                                    HumanMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    AIMessagePromptTemplate,
                                    PromptTemplate)


chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=api_key,
    seed=365,
    max_completion_tokens=100
)


def talk(prompt):
    response = chat.invoke(
        prompt,
    )
    heading2(f"Chat: {prompt}", response.content)

TEMPLATE_S = '{description}'
TEMPLATE_H = '''I've recently adopted a {pet}. 
Could you suggest some {pet} names?'''

message_template_s = SystemMessagePromptTemplate.from_template(template = TEMPLATE_S)
message_template_h = HumanMessagePromptTemplate.from_template(template = TEMPLATE_H)

heading2("System Message Template", message_template_s)


chat_template = ChatPromptTemplate.from_messages([message_template_s, message_template_h])

heading2("Chat Template", chat_template)


chat_value = chat_template.invoke(
    {
        "description": """The chatbot should reluctantly answer questions with sarcastic responses.""",
        "pet": """dog""",
    }
)

heading2("Chat Value", str(chat_value))

talk(chat_value)