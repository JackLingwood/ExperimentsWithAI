# Step 7 in Equipping a chatbot with memory - Summarizing Messages


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
heading(f"{name} LangGraph - Summarizing Messages")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from typing import Literal

class State(MessagesState):
    summary: str

test_state = State()
#test_state["summary"]
# 
bool(test_state.get("summary", ""))

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
    for i in state["messages"]:
        i.pretty_print()
    system_message = f'''
    Here's a quick summary of what's been discussed so far:
    {state.get("summary", "")}

    Keep this in mind as you answer the next question.
    '''
    response = chat.invoke([SystemMessage(system_message)] + state["messages"])
    response.pretty_print()
    return State(messages = [response])

def ask_another_question(state: State) -> State:
    print(f"\n-------> ENTERING ask_another_question:")
    question = "Would you like to ask one more question (yes/no)?"
    print(question)
    return State(messages = [AIMessage(question), HumanMessage(input())])

def summarize_and_delete_messages(state: State) -> State:
    print(f"\n-------> ENTERING trim_messages:")
    new_conversation = ""
    for i in state["messages"]:
        new_conversation += f"{i.type}: {i.content}\n\n"
    summary_instructions = f'''Update the ongoing summary by incorporating the new lines of conversation below.  
Build upon the previous summary rather than repeating it so that the result  
reflects the most recent context and developments.

Previous Summary:
{state.get("summary", "")}

New Conversation:
{new_conversation}
'''
    print(summary_instructions)
    summary = chat.invoke([HumanMessage(summary_instructions)])
    remove_messages = [RemoveMessage(id = i.id) for i in state["messages"][:]]
    return State(messages = remove_messages, summary = summary.content)

def routing_function(state: State) -> Literal["summarize_and_delete_messages", "__end__"]:
    if state["messages"][-1].content == "yes":
        return "summarize_and_delete_messages"
    else:
        return "__end__"

graph = StateGraph(State)
graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("ask_another_question", ask_another_question)
graph.add_node("summarize_and_delete_messages", summarize_and_delete_messages)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "ask_another_question")
graph.add_conditional_edges(source = "ask_another_question", 
                            path = routing_function)
graph.add_edge("summarize_and_delete_messages", "ask_question")

graph_compiled = graph.compile()

note(str(graph_compiled))
print(graph_compiled.get_graph().draw_ascii())

graph_compiled.invoke(State(messages = []))