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
heading(f"{name} Graphing RunnableParallel")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel


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

string_parser = StrOutputParser()

chat_template_books = ChatPromptTemplate.from_template(
    '''
    Suggest three of the best intermediate-level {programming language} books. 
    Answer only by listing the books.
    '''
)

chat_template_projects = ChatPromptTemplate.from_template(
    '''
    Suggest three interesting {programming language} projects suitable for intermediate-level programmers. 
    Answer only by listing the projects.
    '''
)

chain_books = chat_template_books | chat | string_parser

chain_projects = chat_template_projects | chat | string_parser

chain_parallel = RunnableParallel({'books':chain_books, 'projects':chain_projects})

result_parallel = chain_parallel.invoke({'programming language':'Python'})

chain_parallel.get_graph().print_ascii()

result_books = chain_books.invoke({'programming language':'Python'})
result_projects = chain_projects.invoke({'programming language':'Python'})

heading2("Result Books", result_books)
heading2("Result Projects", result_projects)
heading2("Result Parallel", result_parallel)
heading2("Result Parallel Books", result_parallel['books'])
heading2("Result Parallel Projects", result_parallel['projects'])

import json

def prettify_json(data):
    """
    Takes a Python dict or JSON string and returns a pretty-printed JSON string.
    """
    if isinstance(data, str):
        # If input is a JSON string, parse it first
        data = json.loads(data)
    return json.dumps(data, indent=4, ensure_ascii=False)

heading2("Prettified Result Parallel", prettify_json(result_parallel))