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
    max_completion_tokens=500
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

chat_template_time = ChatPromptTemplate.from_template(
     '''
     I'm an intermediate level programmer.
     
     Consider the following literature:
     {books}
     
     Also, consider the following projects:
     {projects}
     
     Roughly how much time would it take me to complete the literature and the projects if I dedicated about 5 hours a day to this?
     
     '''
)

chain_books = chat_template_books | chat | string_parser

chain_projects = chat_template_projects | chat | string_parser

chain_parallel = RunnableParallel({'books':chain_books, 'projects':chain_projects})

result_parallel =  chain_parallel.invoke({'programming language':'Python'})

heading2("chain_parallel", result_parallel)


chain_time1 = (RunnableParallel({'books':chain_books, 
                                'projects':chain_projects}) 
              | chat_template_time 
              | chat 
              | string_parser
             )

chain_time2 = ({'books':chain_books, 
                'projects':chain_projects}
              | chat_template_time 
              | chat 
              | string_parser
             )

# Comparing chain_time1 and chain_time2 you realize
# you do not need to specify RunnableParallel
# in the first case, as the parallelization is already handled by the ChatPromptTemplate.

#heading2("chain_time1", chain_time1.invoke({'programming language':'Python'}))  
print(chain_time2.invoke({'programming language':'Python'}))

chain_time2.get_graph().print_ascii()