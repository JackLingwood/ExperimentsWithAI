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
heading(f"{name} Start LangGraph")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence
from typing import Literal


heading("Starting LangGraph")
# START and END are special nodes in LangGraph
# START is the entry point of the graph, and END is the exit point.
# END is used to indicate the end of the graph.

# TypedDict is used to define the structure of the state = schema.
# It allows you to define the fields and their types in the state.

# Sequence is used to define a sequence of messages.

class State(TypedDict):
    messages: Sequence[BaseMessage]

state = State(messages = [HumanMessage("Could you tell me a grook by Piet Hein?")])

heading2("Initial State", state)

heading2("Messages 0 Plain  - ", state["messages"][0])
heading2("Messages 0 Pretty - ", state["messages"][0].pretty_print())


chat = ChatOpenAI(model = "gpt-4o", 
                  seed = 365, 
                  temperature = 0,
                  api_key=api_key,
                  max_completion_tokens = 100)


#response = chat.invoke(state["messages"])
#heading2("Response from Chat Model", response.pretty_print())

def chatbot(state: State) -> State:    
    print(f"\n-------> ENTERING chatbot:")    
    response = chat.invoke(state["messages"])
    heading3("Response",response.pretty_print())
    return State(messages = [response])

r = chatbot(state)
heading2("Final State", r)

# A StateGraph is a directed graph where nodes are states and edges are transitions between states.
# The graph starts at the START node and ends at the END node.

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
graph_compiled = graph.compile()

heading3("isinstance(graph, Runnable)", isinstance(graph, Runnable))
heading3("isinstance(graph_compiled, Runnable)", isinstance(graph_compiled, Runnable))

heading2("Graph Compiled", graph_compiled)

# -----------------------------------------------
def ask_question(state: State) -> State:    
    print(f"\n-------> ENTERING ask_question:")    
    print("What is your question?")    
    return State(messages = [HumanMessage(input())])

ask_question(State(messages = []))

def ask_another_question(state: State) -> State:
    print(f"\n-------> ENTERING ask_another_question:")
    print("Would you like to ask one more question (yes/no)?")
    return State(messages = [HumanMessage(input())])

ask_another_question(State(messages = []))

def routing_function(state: State) -> str:
    print(f"\n-------> ENTERING routing_function:")    
    
    heading2("state",state)

    if state["messages"][0].content == "yes":
        print("Routing to ask_question")
        return "ask_question"
    else:
        print("Routing to END")
        return "__end__"
    

graph = StateGraph(State)

graph.add_node("ask_question", ask_question)
graph.add_node("chatbot", chatbot)
graph.add_node("ask_another_question", ask_another_question)

graph.add_edge(START, "ask_question")
graph.add_edge("ask_question", "chatbot")
graph.add_edge("chatbot", "ask_another_question")
graph.add_conditional_edges(source = "ask_another_question", 
                            path = routing_function,
                            # path_map={"True": "ask_question", "False": "__end__"}
                            )

graph_compiled = graph.compile()

heading2("Graph Compiled with Conditional Edges", graph_compiled)
#graph_compiled

print(graph_compiled.get_graph().draw_ascii())

graph_compiled.invoke(State(messages = []))