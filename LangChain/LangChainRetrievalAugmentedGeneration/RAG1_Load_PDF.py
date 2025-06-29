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
heading(f"{name} RAG Load a PDF")
# ------------------------------------------------

# pip show langchain

from langchain_community.document_loaders import PyPDFLoader
import copy

loader_pdf = PyPDFLoader("Introduction_to_Data_and_Data_Science.pdf")

pages_pdf = loader_pdf.load()

heading2("Number of pages in PDF", len(pages_pdf))
heading2("First page content", pages_pdf[0].page_content)
heading2("First page metadata", pages_pdf[0].metadata)

heading2("pages_pdf", pages_pdf)

# deepcopy is used to create a copy of the list of pages
# This is necessary to avoid modifying the original list when we remove extra spaces
# We will use this copy to demonstrate how to remove extra spaces from the page content
pages_pdf_cut = copy.deepcopy(pages_pdf)

heading2("pages_pdf_cut", pages_pdf_cut)

# Remove extra spaces from the first page content
' '.join(pages_pdf_cut[0].page_content.split())


heading2("First page content after removing extra spaces", pages_pdf_cut[0].page_content)

for i in pages_pdf_cut:
    i.page_content = ' '.join(i.page_content.split())

heading2("pages_pdf_cut after removing extra spaces", pages_pdf_cut)

heading2("Original document page 0 length", len(pages_pdf[0].page_content))
heading2("Modified document page 0 length", len(pages_pdf_cut[0].page_content))

# https://platform.openai.com/tokenizer