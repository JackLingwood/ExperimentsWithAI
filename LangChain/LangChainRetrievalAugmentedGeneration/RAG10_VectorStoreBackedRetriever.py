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
from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document

embedding = OpenAIEmbeddings(api_key=api_key)

vectorstore = Chroma(persist_directory = "./intro-to-ds-lectures", 
                     embedding_function = embedding)

heading2("Number of documents in the vector store", len(vectorstore.get()['documents']))

# k = 3 means that the retriever will return 3 documents.
# lambda_mult = 0.7 means that the search will be more focused on the most relevant documents, while higher values will increase diversity in the results.
# search_type = 'mmr' means that the retriever will use Maximal Marginal Relevance (MMR) to retrieve documents.
# MMR is a technique that balances relevance and diversity in the retrieved documents.
retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = {'k': 3, 'lambda_mult': 0.7})

heading2("Retriever", retriever)

question = "What software do data scientists use?"
heading2("Question", question)

retrieved_docs = retriever.invoke(question)
heading2("Retrieved Documents", retrieved_docs)

for i in retrieved_docs:
    print(f"Page Content: {i.page_content}\n----------\nLecture Title:{i.metadata['Lecture Title']}\n")