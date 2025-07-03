# Step 4 in Equipping a chatbot with memory - RemoveMessages Class
# This Chatbot uses the RemoveMessages class to remove messages from the conversation history.
# This prevents the conversation history from becoming too long and token heavy.
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
heading(f"{name} LangGraph - The RemoveMessages class")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, add_messages, MessagesState
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, RemoveMessage
from collections.abc import Sequence
from typing import Literal, Annotated

my_list = add_messages([AIMessage("What is your question?"), 
                        HumanMessage("Could you tell me a grook by Piet Hein?"),
                        AIMessage("Certainly! Here's a well-known grook by Piet Hein..."),
                        AIMessage("Would you like to ask one more question?"),
                        HumanMessage("yes"),
                        AIMessage("What is your question?"),
                        HumanMessage("Where was the poet born?"),
                        AIMessage("Piet Hein was born in Copenhagen, Denmark, on December 16, 1905."),
                        AIMessage("Would you like to ask one more question?")],
                       [HumanMessage("yes")]
                      )

heading2("my_list", my_list)


heading2("my_list[:-5]", my_list[:-5])

# Create a list of RemoveMessage objects to remove the last 5 messages
# This is useful to keep the conversation history manageable and prevent it from becoming too long.
remove_messages = [RemoveMessage(id = i.id) for i in my_list[:-5]]

heading2("remove_messages", remove_messages)

heading2("Applying to list","")
print(add_messages(my_list, remove_messages))

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
    note(f"\n-------> ENTERING ask_question: {len(state['messages'])}")
    for i in state["messages"]:
        i.pretty_print()
    
    question = "What is your question?"
    print(question, len(state['messages']), len(state['messages']))
    
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
graph.add_conditional_edges(source = "ask_another_question", 
                            path = routing_function)

graph_compiled = graph.compile()

note(str(graph_compiled))

note("Test 001")

graph_compiled.invoke(MessagesState(messages = []))