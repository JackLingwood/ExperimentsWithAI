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
heading(f"{name} RAG Document Splitting")
# ------------------------------------------------


# Splitting by a predefined chunk size
# Chunk overlap of 50 characters is used to ensure that the chunks are not too similar
# and to maintain some context between chunks.




from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter

loader = Docx2txtLoader("Introduction_to_Data_and_Data_Science.docx")
pages = loader.load()

def report(title, doc_pages):
    heading2(title, "")
    heading3("First page metadata", doc_pages[0].metadata)
    heading3("Number of pages in DOCX", len(doc_pages))
    first_page_content_length = len(doc_pages[0].page_content)
    heading3("First page content length", first_page_content_length)
    
    total_chars = sum(len(page.page_content) for page in doc_pages)
    heading3("Total number of characters in document", total_chars)
    last_page_content_length = len(doc_pages[-1].page_content)
    heading3("Last page content length", last_page_content_length)

def reportSplitter(splitter):
    heading2("Character Text Splitter", "")
    heading3("Separator", splitter._separator)
    heading3("Chunk size", splitter._chunk_size)
    heading3("Chunk overlap", splitter._chunk_overlap)

report("Initial report on loaded DOCX", pages)

for i in range(len(pages)):
    pages[i].page_content = ' '.join(pages[i].page_content.split())

report("Report after removing extra spaces", pages)

page_chunk_size = 500
page_overlap = 0

char_splitter = CharacterTextSplitter(separator = "", 
                                      chunk_size = 500, 
                                      chunk_overlap = 0)



reportSplitter(char_splitter)

pages_char_split = char_splitter.split_documents(pages)

report("Report after first character splitting", pages_char_split)

char_splitter2 = CharacterTextSplitter(separator = "", 
                                      chunk_size = 500, 
                                      chunk_overlap = 50)

reportSplitter(char_splitter2)

pages_char_split2 = char_splitter.split_documents(pages)

report("Report after second character splitting with overlap", pages_char_split2)

char_splitter3 = CharacterTextSplitter(separator = ".", 
                                      chunk_size = 500, 
                                      chunk_overlap = 50)

reportSplitter(char_splitter3)

pages_char_split3 = char_splitter3.split_documents(pages)

report("Report after third character splitting with period as separator", pages_char_split3)