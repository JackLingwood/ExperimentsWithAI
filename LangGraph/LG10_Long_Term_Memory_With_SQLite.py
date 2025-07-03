# Long-term memory with SQLite
# This example demonstrates how to use LangGraph with SQLite to maintain long-term memory.
# The chatbot will remember the context of the conversation across multiple interactions,
# allowing it to provide more relevant and context-aware responses.
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
heading(f"{name} LangGraph - Long-term memory with SQLite")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

class State(MessagesState):
    summary: str

chat = ChatOpenAI(model = "gpt-4o", 
                  seed = 365, 
                  temperature = 0, 
                  max_completion_tokens = 100,
                  api_key = api_key)

def ask_question(state: State) -> State:
    print(f"\n-------> ENTERING ask_question:")
    question = "What is your question?"
    print(question)
    return State(messages = [AIMessage(question), HumanMessage(input())])

def chatbot(state: State) -> State:
    print(f"\n-------> ENTERING chatbot:")
    system_message = f'''
    Here's a quick summary of what's been discussed so far:
    {state.get("summary", "")}

    Keep this in mind as you answer the next question.
    '''
    response = chat.invoke([SystemMessage(system_message)] + state["messages"])
    response.pretty_print()
    return State(messages = [response])

def summarize_messages(state: State) -> State:
    print(f"\n-------> ENTERING summarize_messages:")

    new_conversation = ""
    for i in state["messages"]:
        new_conversation += f"{i.type}: {i.content}\n\n"
    
    summary_instructions = f'''
Update the ongoing summary by incorporating the new lines of conversation below. 
Build upon the previous summary rather than repeating it, 
so that the result reflects the most recent context and developments.
Respond only with the summary.

Previous Summary:
{state.get("summary", "")}

New Conversation:
{new_conversation}
'''
    print(summary_instructions)

    summary = chat.invoke([HumanMessage(summary_instructions)])
    
    remove_messages = [RemoveMessage(id = i.id) for i in state["messages"][:]]

    return State(messages = remove_messages, summary = summary.content)

graph = StateGraph(State)
graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("summarize_messages", summarize_messages)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "summarize_messages")
graph.add_edge("summarize_messages", END)

db_path =  os.getcwd() + "/long_term_memory.db"
con = sqlite3.connect(database = db_path, check_same_thread = False)

checkpointer = SqliteSaver(con)

graph_compiled = graph.compile(checkpointer)
print(graph_compiled.get_graph().draw_ascii())

config1 = {"configurable": {"thread_id": "1"}}

graph_compiled.invoke(State(), config1)

# In production use postgresql or redis
