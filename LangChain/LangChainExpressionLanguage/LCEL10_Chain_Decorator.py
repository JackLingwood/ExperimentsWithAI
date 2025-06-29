from dotenv import load_dotenv
import os
import sys

# Add Shared folder to sys.path
sys.path.append(os.path.abspath("Shared"))

from utils import heading, heading2, heading3, clearConsole

clearConsole()

# Headings
name = os.path.basename(__file__)
heading(f"{name} @Chain decorator")

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain

def find_sum(x):
    return sum(x)

def find_square(x):
    return x**2

chain1 = RunnableLambda(find_sum) | RunnableLambda(find_square)

result1 = chain1.invoke([1, 2, 5])
heading2("chain1", result1)

@chain
def runnable_sum(x):
    return sum(x)

@chain
def runnable_square(x):
    return x**2

type(runnable_sum), type(runnable_square)
heading2("type(runnable_sum)", type(runnable_sum))
heading2("type(runnable_square)", type(runnable_square))

chain2 = runnable_sum | runnable_square
result2 = chain2.invoke([1, 2, 5])
heading2("chain2", result2)