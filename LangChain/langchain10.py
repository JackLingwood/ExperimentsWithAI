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
heading(f"{name} Comma-separated list output parser")

api_key = os.environ.get("api_key")


from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.output_parsers import DatetimeOutputParser


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



message_h = HumanMessage(content = f'''When was the Danish poet Piet Hein born?
{DatetimeOutputParser().get_format_instructions()}
''')

heading2("Message_h.content", message_h.content)


response = chat.invoke([message_h])
heading2("response.content",response.content)

date_output_parser = DatetimeOutputParser()

heading2("Date Output Parser Instructions", date_output_parser.get_format_instructions())

response_parsed = date_output_parser.invoke(response)

heading2("Parsed Response", response_parsed)