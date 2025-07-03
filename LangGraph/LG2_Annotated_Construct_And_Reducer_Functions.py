# Step 1 in Equipping a chatbot with memory

# ---------------------------------------------------------------------------------------------
from dotenv import load_dotenv
import os
import sys

# Add Shared folder to sys.path
sys.path.append(os.path.abspath("Shared"))

from utils import heading, heading2, heading3, clearConsole, note
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
heading(f"{name} LangGraph - Annotated Construct And Reducer Functions")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, add_messages
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from collections.abc import Sequence
from typing import Literal, Annotated

my_list = add_messages(
    [HumanMessage("Hi! I'm Oscar."), AIMessage("Hey, Oscar. How can I assist you?")],
    [HumanMessage("Could you summarize today's news?")],
)

heading2("my_list", my_list)

# Annotated allows you to add metadata to the type, which can be used by LangGraph to process the messages correctly.
# In this case, we are using add_messages to annotate the messages with the correct type.

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] # This allows LangGraph to understand that messages is a sequence of messages.