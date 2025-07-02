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
#from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

vectorstore = Chroma(persist_directory = "./intro-to-ds-lectures", 
                     embedding_function = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key))

heading2("Number of documents in the vector store",len(vectorstore.get()['documents']))

retriever = vectorstore.as_retriever(search_type = 'mmr', search_kwargs = {'k':3, 'lambda_mult':0.7})

TEMPLATE = '''
Answer the following question:
{question}

To answer the question, use only the following context:
{context}

#At the end of the response, specify the name of the lecture this context is taken from in the format:
#Resources: *Lecture Title*
#where *Lecture Title* should be substituted with the title of all resource lectures.
'''

prompt_template = PromptTemplate.from_template(TEMPLATE)
heading2("Prompt Template", prompt_template)


chat = ChatOpenAI(model_name = 'gpt-4', 
                  seed = 365,
                  #model_kwargs = {'seed':365},
                  max_tokens = 250,
                  api_key=api_key)

question = "What software do data scientists use?"
heading2("Question", question)

chain = {'context': retriever, 'question': RunnablePassthrough()} | prompt_template
r = chain.invoke(question)
heading2("chain.invoke(question) (r) =", r)

heading2("r['text']", r.text)


retrieved_docs = retriever.invoke(question)
heading2("Retrieved Documents", retrieved_docs)

heading("Final Output")

for i in retrieved_docs:
    print('─' * 150)
    heading2("Page Content", i.page_content)
    heading2("Lecture Title", i.metadata['Lecture Title'])    