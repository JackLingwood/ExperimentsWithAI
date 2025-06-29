# ------------------------------------------------
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
heading(f"{name} RAG Load a DOCX")
# ------------------------------------------------

from langchain_community.document_loaders import Docx2txtLoader


loader_docx = Docx2txtLoader("Introduction_to_Data_and_Data_Science.docx")

pages_docx = loader_docx.load()

heading2("Number of pages in DOCX", len(pages_docx))
heading2("First page content", pages_docx[0].page_content)
heading2("First page metadata", pages_docx[0].metadata)
heading2("pages_docx", pages_docx)


