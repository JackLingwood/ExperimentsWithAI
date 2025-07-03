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
heading(f"{name} Vector Store Backed Retriever with MMR")
api_key = os.environ.get("api_key")
# ---------------------------------------------------------------------------------------------

from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

vectorstore = Chroma(persist_directory = "./intro-to-ds-lectures", 
                     embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key))

length = len(vectorstore.get()['documents'])
heading2("Number of documents in the vector store", length)

retriever = vectorstore.as_retriever(search_type = 'mmr', 
                                     search_kwargs = {'k':3, 
                                                      'lambda_mult':0.7})


TEMPLATE = '''
Answer the following question:
{question}

To answer the question, use only the following context:
{context}

At the end of the response, specify the name of the lecture this context is taken from in the format:
Resources: *Lecture Title*
where *Lecture Title* should be substituted with the title of all resource lectures.
'''

prompt_template = PromptTemplate.from_template(TEMPLATE)

chat = ChatOpenAI(model_name = 'gpt-4', 
                  model_kwargs = {'seed':365},
                  max_tokens = 250,
                  api_key=api_key)

question = "What software do data scientists use?"

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | chat
    | StrOutputParser()
)

heading2("Question", question)
heading("Final Output")
print(chain.invoke(question))

# Stuffing the documents into the response means that the response will include the content of the documents retrieved by the retriever.
# The opposite of stuffing is summarization, where the response will include a summary of the documents retrieved by the retriever.
# Document refinement is a technique that can be used to improve the quality of the documents retrieved by the retriever.
# This can be done by applying a series of transformations to the documents, such as removing duplicates, correcting spelling errors, and so on.
