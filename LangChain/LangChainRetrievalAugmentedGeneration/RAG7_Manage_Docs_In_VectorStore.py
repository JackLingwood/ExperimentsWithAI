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
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document

embedding = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=api_key)

vectorstore_from_directory = Chroma(persist_directory = "./intro-to-ds-lectures", 
                                    embedding_function = embedding)

r0 = vectorstore_from_directory.get()
heading2("r0 - All documents ",r0)

some_document_id  = r0["ids"][0]
heading3("some_document_id", some_document_id)

r1 = vectorstore_from_directory.get(some_document_id)
heading2("r1 - Get one specific known document", r1)


r2 = vectorstore_from_directory.get(ids = some_document_id, 
                               include = ["embeddings"])
heading2("r2 - One specific existing document with embedings", r2)


added_document = Document(page_content='Alright! So… Let’s discuss the not-so-obvious differences between the terms analysis and analytics. Due to the similarity of the words, some people believe they share the same meaning, and thus use them interchangeably. Technically, this isn’t correct. There is, in fact, a distinct difference between the two. And the reason for one often being used instead of the other is the lack of a transparent understanding of both. So, let’s clear this up, shall we? First, we will start with analysis', 
                          metadata={'Course Title': 'Introduction to Data and Data Science', 
                                    'Lecture Title': 'Analysis vs Analytics'})

new_id = vectorstore_from_directory.add_documents([added_document])[0]
heading2("New Document ID", new_id)



r3 = vectorstore_from_directory.get()
heading2("r3 - All documents", r3)

r4 = vectorstore_from_directory.get(new_id)
heading2("r4 - the new document", r4)



updated_document = Document(page_content='Great! We hope we gave you a good idea about the level of applicability of the most frequently used programming and software tools in the field of data science. Thank you for watching!', 
                            metadata={'Course Title': 'Introduction to Data and Data Science', 
                                     'Lecture Title': 'Programming Languages & Software Employed in Data Science - All the Tools You Need'})

r5 = vectorstore_from_directory.update_document(document_id = new_id, 
                                           document = updated_document)
heading2("r5 - No respoonse expected", r5)

r6 = vectorstore_from_directory.get(new_id)
heading2("r6 - the new document", r6)

r7 = vectorstore_from_directory.delete(new_id)
heading2("Document Deleted R7", r7)

r8 = vectorstore_from_directory.get(new_id)
heading2("r8 - the specific single document after it was deleted", r8)