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
# Get name of the current file
name = os.path.basename(__file__)
heading(f"{name} Human & System Message with LangChain OpenAI Chat")
api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

message_human = HumanMessage(
    content="I've recently adopted a dog. Could you suggest some dog names?"
)


message_system = SystemMessage(
    content='''You are Marv, a chatbot that reluctantly answers questions with sarcastic responses.'''
)

talk([message_system,message_human])