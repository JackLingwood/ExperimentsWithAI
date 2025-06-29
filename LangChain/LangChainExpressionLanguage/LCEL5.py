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
heading(f"{name} Runnables")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


# The runnable class is a callable that can be used to invoke the chat template.
# Prompt Template, Chat Model and Output Parser are all runnables.

# RunnablePassthrough is a runnable that simply passes the input to the output without any modification.
result = RunnablePassthrough().invoke([1, 2, 3])
heading2("result", result)

chat_template_tools = ChatPromptTemplate.from_template('''
What are the five most important tools a {job title} needs?
Answer only by listing the tools.
''')

chat_template_strategy = ChatPromptTemplate.from_template('''
Considering the tools provided, develop a strategy for effectively learning and mastering them:
{tools}
''')

heading2("chat_template_tools", chat_template_tools)
heading2("chat_template_strategy", chat_template_strategy)

string_parser = StrOutputParser()

chain_tools = (chat_template_tools | chat | string_parser | {'tools':RunnablePassthrough()})
chain_strategy = chat_template_strategy | chat | string_parser

heading("Chain Tools")
print(chain_tools.invoke({'job title':'data scientist'}))

heading("Chain Strategy")
print(chain_strategy.invoke({'tools':'''
1. Python
2. R Programming
3. SQL
4. Tableau
5. Hadoop
'''}))

chain_combined = chain_tools | chain_strategy

heading("Chain Combined")
print(chain_combined.invoke({'job title':'data scientist'}))

chain_long = (
    chat_template_tools
    | chat
    | string_parser
    | {"tools": RunnablePassthrough()}
    | chat_template_strategy
    | chat
    | string_parser
)
