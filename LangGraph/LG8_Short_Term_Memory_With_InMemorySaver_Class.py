# Step 8 in Equipping a chatbot with memory - Short Term Memory with InMemorySaver Class
# A checkpointing class that saves the state of the conversation in memory.
# This allows the chatbot to maintain a short-term memory of the conversation.
# The StateSnapshot class is used to save the state of the conversation.
# This is useful for short-term memory, where the chatbot needs to remember the context of the
# conversation for a limited time.
# Threads are a collection of checkpoints that can be used to manage the state of the conversation.
# This allows the chatbot to maintain a short-term memory of the conversation.
# Short-Term Memory is thread-specific, meaning that each thread has its own memory.


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
heading(f"{name} LangGraph - Short Term Memory with InMemorySaver Class")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver

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

checkpointer = InMemorySaver()
graph_compiled = graph.compile(checkpointer)
print(graph_compiled.get_graph().draw_ascii())

config1 = {"configurable": {"thread_id": "1"}}
config2 = {"configurable": {"thread_id": "2"}}

graph_compiled.invoke(State(), config1)
graph_compiled.invoke(State(), config1)
graph_compiled.invoke(State(), config1)
graph_compiled.invoke(State(), config2)