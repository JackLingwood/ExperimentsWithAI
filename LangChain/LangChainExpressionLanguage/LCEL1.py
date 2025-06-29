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
client = OpenAI(api_key=api_key)



from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

list_instructions = CommaSeparatedListOutputParser().get_format_instructions()

list_instructions

chat_template = ChatPromptTemplate.from_messages([
    ('human', 
     "I've recently adopted a {pet}. Could you suggest three {pet} names? \n" + list_instructions)])

print(chat_template.messages[0].prompt.template)

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

list_output_parser = CommaSeparatedListOutputParser()

chat_template_result = chat_template.invoke({'pet':'dog'})

chat_result = chat.invoke(chat_template_result)
heading2("Chat Result", chat_result.content)

list_output_parser.invoke(chat_result)

chain = chat_template | chat

heading2("Chain", chain)
heading2("Chain.invoke", chain.invoke({'pet':'dog'}))