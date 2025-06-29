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

chain.invoke({'pet':'dog', 'breed':'shepherd'})

# Timing batch
import time
start = time.time()
chain.batch([{'pet':'dog', 'breed':'shepherd'}, 
             {'pet':'dragon', 'breed':'night fury'}])
end = time.time()
print(f"Batch elapsed time: {end - start:.2f} seconds")

# Timing single invocation
start = time.time()
chain.invoke({'pet':'dog', 'breed':'shepherd'})
end = time.time()
print(f"Single invoke (dog) elapsed time: {end - start:.2f} seconds")

start = time.time()
chain.invoke({'pet':'dragon', 'breed':'night fury'})
end = time.time()
print(f"Single invoke (dragon) elapsed time: {end - start:.2f} seconds")

response = chain.stream({'pet':'dragon', 'breed':'night fury'})

heading2("response", response)

# Generator functions in Python allow for defining functions that behave like iterators, allowing us to loop over their ouput.
# This is useful for streaming responses, as it allows us to process each part of the response as it comes in.
# The next() function retrieves the next item from the iterator, which in this case is the
# next part of the streamed response.
# The response is a generator, so we can iterate over it to get the content.
# If you want to print the content of the response, you can use a for loop to
# iterate over the response and print each part.

# use next() to get the first part of the response

next(response)

for i in response:
    print(i.content, end = '')