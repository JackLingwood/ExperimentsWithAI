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
heading(f"{name} Few Shot Prompt Template")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)


from langchain_openai.chat_models import ChatOpenAI


from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate, 
                                    AIMessagePromptTemplate, 
                                    FewShotChatMessagePromptTemplate)



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


TEMPLATE_H = '''I've recently adopted a {pet}. 
Could you suggest some {pet} names?'''
TEMPLATE_AI = '''{response}'''

message_template_h = HumanMessagePromptTemplate.from_template(template = TEMPLATE_H)
message_template_ai = AIMessagePromptTemplate.from_template(template = TEMPLATE_AI)

example_template = ChatPromptTemplate.from_messages([message_template_h, 
                                                     message_template_ai])

examples = [{'pet':'dog', 
             'response':'''Oh, absolutely. Because nothing screams "I'm a responsible pet owner" 
like asking a chatbot to name your new furball. How about "Bark Twain" (if it's a literary hound)? '''}, 
            
            {'pet':'cat', 
             'response':'''Oh, absolutely. Because nothing screams "I'm a unique and creative individual" 
             like asking a chatbot to name your cat. How about "Furry McFurFace", "Sir Meowsalot", or "Catastrophe"? '''}, 
            
            {'pet':'fish', 
             'response':
             '''Oh, absolutely. Because nothing screams "I'm a fun and quirky pet owner" 
             like asking a chatbot to name your fish. How about "Fin Diesel", "Gill Gates", or "Bubbles"?'''}]

few_shot_prompt = FewShotChatMessagePromptTemplate(examples = examples, 
                                                   example_prompt = example_template, 
                                                   input_variables = ['pet'])

chat_template = ChatPromptTemplate.from_messages([few_shot_prompt, 
                                                  message_template_h])

chat_value = chat_template.invoke({'pet':'rabbit'})

heading2("Chat Value", chat_value)

heading2("Chat Value Messages","")

for i in chat_value.messages:
    heading3(f'{i.type} Message', i.content)
    print()


talk(chat_value)


# END