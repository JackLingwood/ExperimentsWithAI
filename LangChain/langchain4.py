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
heading(f"{name} Human & AI Messages with LangChain OpenAI Chat - Few-shot Prompting")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

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

message_human_dog = HumanMessage(
    content="I've recently adopted a dog. Could you suggest some dog names?"
)


message_AI_dog = AIMessage(
    content=''' Oh, absolutely. Because nothing screams "I'm a responsible pet owner" like asking a chatbot to name your new furball. How about "Bark Twain" if it's a literary hound, or "Sir Wag-a-lot" for the knightly type? Maybe "Bark Zuckerberg" if it's into social media, or "Biscuit" if it's as bland as your decision-making skills.'''
)

message_AI_cat = AIMessage(
    content=''' Oh, absolutely. Because nothing screams "I'm a responsible pet owner" like asking a chatbot to name your new furball. How about "Bark Twain" if it's a literary hound, or "Sir Wag-a-lot" for the knightly type? Maybe "Bark Zuckerberg" if it's into social media, or "Biscuit" if it's as bland as your decision-making skills.'''
)

message_human_cat = HumanMessage(
    content="I've recently adopted a cat. Could you suggest some cat names?"
)

message_human_dog = HumanMessage(content = ''' I've recently adopted a dog. Can you suggest some dog names? ''')
message_ai_dog = AIMessage(content = ''' Oh, absolutely. Because nothing screams "I'm a responsible pet owner" 
like asking a chatbot to name your new furball. How about "Bark Twain" (if it's a literary hound)? ''')

message_human_cat = HumanMessage(content = ''' I've recently adopted a cat. Can you suggest some cat names? ''')
message_ai_cat = AIMessage(content = ''' Oh, absolutely. Because nothing screams "I'm a unique and creative individual" 
like asking a chatbot to name your cat. How about "Furry McFurFace", "Sir Meowsalot", or "Catastrophe"? ''')

message_human_fish = HumanMessage(content = ''' I've recently adopted a fish. Can you suggest some fish names? ''')

# Few-shot prompting is a technique where you provide the model with examples of the kind of responses you want it to generate.
# This can help guide the model's behavior and improve the quality of its responses.
# Here were using few-shot prompting to guide the model's responses for both dog and cat names.
talk([message_human_dog, message_AI_dog, message_human_cat, message_AI_cat, message_human_fish])
