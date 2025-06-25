import os
import sys

sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole

clearConsole()

# Change current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())


import config
api_key = config.api_key

os.environ["USER_AGENT"] = "MyLangChainApp/1.0"


heading("LangChain")

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

url = "https://365datascience.com/upcoming-courses"

loader = WebBaseLoader(url)

raw_documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings(openai_api_key=api_key)

vectorstore = FAISS.from_documents(documents, embeddings)


memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(openai_api_key=api_key, 
                                                  model="gpt-3.5-turbo", 
                                                  temperature=0), 
                                           vectorstore.as_retriever(), 
                                           memory=memory)

query = "What is the next course to be uploaded on the 365DataScience platform?"

heading2("Query", query)


result = qa({"question": query})

heading2("Result", result['answer'])

