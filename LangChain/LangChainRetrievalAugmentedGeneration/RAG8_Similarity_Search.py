# ------------------------------------------------
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
heading(f"{name} Managing Documents in a Chroma Vector Store")
api_key = os.environ.get("api_key")
# ------------------------------------------------

from langchain_openai.embeddings import OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document


embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)

vectorstore = Chroma(persist_directory = "./intro-to-ds-lectures", 
                     embedding_function = embedding)

added_document = Document(page_content='Alright! So… How are the techniques used in data, business intelligence, or predictive analytics applied in real life? Certainly, with the help of computers. You can basically split the relevant tools into two categories—programming languages and software. Knowing a programming language enables you to devise programs that can execute specific operations. Moreover, you can reuse these programs whenever you need to execute the same action', 
                         metadata={'Course Title': 'Introduction to Data and Data Science', 
                                  'Lecture Title': 'Programming Languages & Software Employed in Data Science - All the Tools You Need'})

vectorstore.add_documents([added_document])

question = "What programming languages do data scientists use?"

heading2("Question", question)

# k = 5, get at most 5 documents
retrieved_docs = vectorstore.similarity_search(query = question, k = 5)

heading2("Number of retrieved documents", len(retrieved_docs))

heading2("Details of retrieved documents","")

for i in retrieved_docs:
    print(f"Page Content: {i.page_content}\n----------\nLecture Title:{i.metadata['Lecture Title']}\n")