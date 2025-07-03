# Step 5 in Equipping a chatbot with memory - Trimming messages

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
heading(f"{name} LangGraph - Trimming Messages")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, add_messages, MessagesState
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, RemoveMessage
from collections.abc import Sequence
from typing import Literal, Annotated

chat = ChatOpenAI(model = "gpt-4o", 
                  seed = 365, 
                  temperature = 0, 
                  max_completion_tokens = 100,
                  api_key = api_key)

def ask_question(state: MessagesState) -> MessagesState:
    
    note(f"\n-------> ENTERING ask_question: {len(state['messages'])}")
    for i in state["messages"]:
        i.pretty_print()
    
    question = "What is your question?"
    print(question, len(state['messages']))
    
    return MessagesState(messages = [AIMessage(question), HumanMessage(input())])

def chatbot(state: MessagesState) -> MessagesState:
    
    note(f"\n-------> ENTERING chatbot: {len(state['messages'])}")

    for i in state["messages"]:
        i.pretty_print()
    
    response = chat.invoke(state["messages"])
    response.pretty_print()
    
    return MessagesState(messages = [response])

def ask_another_question(state: MessagesState) -> MessagesState:
    
    note(f"\n-------> ENTERING ask_another_question: {len(state['messages'])}")

    for i in state["messages"]:
        i.pretty_print()
    
    question = "Would you like to ask one more question (yes/no)?"
    print(question, len(state['messages']))
    
    return MessagesState(messages = [AIMessage(question), HumanMessage(input())])

def trim_messages(state: MessagesState) -> MessagesState:
    note(f"\n-------> ENTERING trim_messages: {len(state['messages'])}")
    
    remove_messages = [RemoveMessage(id = i.id) for i in state["messages"][:-5]]
    
    return MessagesState(messages = remove_messages)

def routing_function(state: MessagesState) -> Literal["trim_messages", "__end__"]:
    
    if state["messages"][-1].content == "yes":
        return "trim_messages"
    else:
        return "__end__"

graph = StateGraph(MessagesState)

graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("ask_another_question", ask_another_question)
graph.add_node("trim_messages", trim_messages)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "ask_another_question")
graph.add_conditional_edges(source = "ask_another_question", 
                            path = routing_function)
graph.add_edge("trim_messages", "ask_question")

graph_compiled = graph.compile()

note(str(graph_compiled))
print(graph_compiled.get_graph().draw_ascii())

graph_compiled.invoke(MessagesState(messages = []))