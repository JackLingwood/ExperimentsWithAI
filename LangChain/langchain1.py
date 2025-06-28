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
heading("Langchain 1")
heading2("Environment Variables  and Configurations", "")
for key, value in os.environ.items():
    if "KEY" in key.upper():
        heading3(key, value)

# OpenAI API setup
api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)


basic_messages = messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
]

# Sarcastic chatbot messages
sarcastic_messages = [
    {
        'role': 'system',
        'content': (
            "You are Marv, a chatbot that reluctantly answers questions with sarcastic responses."
        )
    },
    {
        'role': 'user',
        'content': (
            "I've recently adopted a dog. Could you suggest some dog names?"
        )
    }
]

# Example completion (not using sarcastic_messages yet)
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=sarcastic_messages,
    max_tokens=100,
    temperature=0.2,
)

heading2("completion", completion)

def showResponse(x):
    heading2("Response from OpenAI API",x.choices[0].message.content)

showResponse(completion)


completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": """ Could you explain briefly what a black hole is? """,
        }
    ],
    max_tokens=250,
    temperature=0,
)

showResponse(completion)


heading("Streaming Response from OpenAI API")

stream_completion = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {
            "role": "user",
            "content": """ Could you explain briefly what a black hole is? """,
        }
    ],
    max_tokens=250,
    temperature=0,
    seed=365,
    stream=True
)

# for chunk in stream_completion:
#     print(chunk.choices[0].delta.content, end="")


for i in stream_completion:
    print(i.choices[0].delta.content, end = "")
