# Step 3 in Equipping a chatbot with memory - Message State Class
# This Chatbot uses the MessageState class to manage the state of the conversation.
# This allows the chatbot to keep track of the messages exchanged during the conversation.
# This approach is suitable for short conversations only because it becomes token heavy for longer conversations.
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
heading(f"{name} LangGraph - The MessageState class")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, add_messages, MessagesState
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from collections.abc import Sequence
from typing import Literal, Annotated

my_list = add_messages([HumanMessage("Hi! I'm Oscar."), 
                        AIMessage("Hey, Oscar. How can I assist you?")],
                       [HumanMessage("Could you summarize today's news?")])

heading2("my_list", my_list)

chat = ChatOpenAI(model = "gpt-4o", 
                  seed = 365, 
                  temperature = 0, 
                  max_completion_tokens = 100,
                  api_key = api_key)

def ask_question(state: MessagesState) -> MessagesState:
    print(f"\n-------> ENTERING ask_question:")
    for i in state["messages"]:
        i.pretty_print()
    question = "What is your question?"
    print(question)
    return MessagesState(messages = [AIMessage(question), HumanMessage(input())])

def chatbot(state: MessagesState) -> MessagesState:
    print(f"\n-------> ENTERING chatbot:")
    for i in state["messages"]:
        i.pretty_print()
    response = chat.invoke(state["messages"])
    response.pretty_print()
    return MessagesState(messages = [response])

def ask_another_question(state: MessagesState) -> MessagesState:
    print(f"\n-------> ENTERING ask_another_question:")
    for i in state["messages"]:
        i.pretty_print()
    
    question = "Would you like to ask one more question (yes/no)?"
    print(question)
    
    return MessagesState(messages = [AIMessage(question), HumanMessage(input())])

def routing_function(state: MessagesState) -> Literal["ask_question", "__end__"]:
    if state["messages"][-1].content == "yes":
        return "ask_question"
    else:
        return "__end__"

graph = StateGraph(MessagesState)

graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("ask_another_question", ask_another_question)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "ask_another_question")
graph.add_conditional_edges(source = "ask_another_question", path = routing_function)

graph_compiled = graph.compile()

note(str(graph_compiled))

graph_compiled.invoke(MessagesState(messages = []))