import os
import sys

sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole

clearConsole()

# Change current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())


import config
heading("ChatGPT API")
api_key = config.api_key
heading2("api_key", api_key)

import openai

# Replace with your actual OpenAI API key
#api_key = "sk-..."  

client = openai.OpenAI(api_key=api_key)

def chatgpt_response(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    reply = chatgpt_response(user_prompt)
    print("ChatGPT:", reply)