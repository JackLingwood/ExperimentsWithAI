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
from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)

vectorstore = Chroma(persist_directory = "./intro-to-ds-lectures", 
                     embedding_function = embedding)

# added_document = Document(page_content='Alright! So… How are the techniques used in data, business intelligence, or predictive analytics applied in real life? Certainly, with the help of computers. You can basically split the relevant tools into two categories—programming languages and software. Knowing a programming language enables you to devise programs that can execute specific operations. Moreover, you can reuse these programs whenever you need to execute the same action', 
#                         metadata={'Course Title': 'Introduction to Data and Data Science', 
#                                   'Lecture Title': 'Programming Languages & Software Employed in Data Science - All the Tools You Need'})

# vectorstore.add_documents([added_document])

question = "What software do data scientists use?"

# lambda_mult = 1 means that the search will be more focused on the most relevant documents, while higher values will increase diversity in the results.

retrieved_docs = vectorstore.max_marginal_relevance_search(
    query=question, 
    k=3, 
    lambda_mult = 0.7, 
    filter = {"Lecture Title": "Programming Languages & Software Employed in Data Science - All the Tools You Need"}
)

# lambda_mult = closer to 0 means that the search will focus on document diversity
# lambda_mult = 1 means that the search will be more focused on the most relevant documents, while higher values will increase diversity in the results.


for i in retrieved_docs:
    print(f"Page Content: {i.page_content}\n----------\nLecture Title:{i.metadata['Lecture Title']}\n")



# Maximal Marginal Relevance (MMR) is a technique used to retrieve documents that are both relevant to a query and diverse from each other. It helps in avoiding redundancy in the retrieved documents, ensuring that the results cover a broader range of information while still being pertinent to the user's query.
# MMR works by balancing the relevance of documents to the query with their diversity from each other. The algorithm selects documents that are not only relevant but also maximally different from the already selected documents, thus providing a more comprehensive set of results.
# In this example, we use the `max_marginal_relevance_search` method from the Chroma vector store to perform a search that retrieves documents relevant to the query while ensuring diversity among the results.
# Note: The `lambda_mult` parameter controls the trade-off between relevance and diversity. A higher value increases diversity, while a lower value focuses more on relevance.
# Note: The `filter` parameter allows you to filter the results based on specific metadata, such as the lecture title in this case.
# This example demonstrates how to use MMR to retrieve diverse and relevant documents from a vector store, enhancing the quality of search results in applications like question-answering systems or information retrieval tasks.
# Note: Ensure that the Chroma vector store is properly set up and contains the necessary documents before running this code.
# Note: The `max_marginal_relevance_search` method is a powerful tool for retrieving diverse and relevant documents, making it suitable for applications like question-answering systems or information retrieval tasks.
