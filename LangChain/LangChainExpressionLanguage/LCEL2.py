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


# START

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chat_template = ChatPromptTemplate.from_messages([
    ('human', 
     "I've recently adopted a {pet} which is a {breed}. Could you suggest several training tips?")])

chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=api_key,
    seed=365,
    max_completion_tokens=100
)

chain = chat_template | chat

result = chain.invoke({'pet':'dog', 'breed':'shepherd'})
heading2("Result for Dog Shepherd", result.content)


# Batching multiple prompts

# Write ipython's %%time in regular python


import time

start = time.time()
# Your code block here
result2 = chain.batch(
    [{"pet": "dog", "breed": "shepherd"}, {"pet": "dragon", "breed": "night fury"}]
)
end = time.time()
heading2("Batch Result", f"Elapsed time: {end - start:.2f} seconds")

heading2("Batch Result2[0]", result2[0].content)
heading2("Batch Result2[1]", result2[1].content)


# Timing single invocations
start = time.time()
chain.invoke({'pet':'dog', 'breed':'shepherd'})
chain.invoke({'pet':'dragon', 'breed':'night fury'})
end = time.time()
heading2("Double Invoke",f"Elapsed time: {end - start:.2f} seconds")

# END
