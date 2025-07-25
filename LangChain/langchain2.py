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
heading("Langchain 2")
api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model_name="gpt-4",
    temperature=0,
    openai_api_key=api_key,
    seed=365,
    max_completion_tokens=100,
    #model_kwargs={"seed": 365},
)


def talk(prompt):
    response = chat.invoke(
        prompt,
    )
    heading2(f"Chat: {prompt}", response.content)


talk('''I've recently adopted a dog. Could you suggest some dog names?''')
talk('''Is Juju a good name for a dog?''')
