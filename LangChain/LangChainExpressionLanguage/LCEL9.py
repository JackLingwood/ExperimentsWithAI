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
heading(f"{name} RunnableLambda")

api_key = os.environ.get("api_key")
client = OpenAI(api_key=api_key)

from langchain_core.runnables import RunnableLambda

find_sum = lambda x: sum(x)
heading2("find_sum", find_sum([1, 2, 5]))

find_square = lambda x: x**2
heading2("find_square", find_square(8))

runnable_sum = RunnableLambda(lambda x: sum(x))
result1 = runnable_sum.invoke([1, 2, 5])
heading2("runnable_sum", result1)

runnable_square = RunnableLambda(lambda x: x**2)
result2 = runnable_square.invoke(8)
heading2("runnable_square", result2)

chain = runnable_sum | runnable_square
result3 = chain.invoke([1, 2, 5])
heading2("chain", result3)

heading2("Graphing RunnableParallel","")
chain.get_graph().print_ascii()

# Lambda functions can be used to create Runnables that can be used in a RunnableParallel.
# Lambda functions are anonymous functions that can be used to create simple Runnables.