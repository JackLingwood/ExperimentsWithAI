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
heading(f"{name} RAG Document Splitting with Markdown")
# ------------------------------------------------


from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter

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

def remove_extra_spaces_from_pages(pages):
    """
    Remove extra spaces from the page content of each document in the pages list.
    """
    for i in range(len(pages)):
        # Replace multiple spaces with a single space
        pages[i].page_content = ' '.join(pages[i].page_content.split())
    return pages

remove_extra_spaces_from_pages(pages)


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


loader_docx2 = Docx2txtLoader("Introduction_to_Data_and_Data_Science_2.docx")
pages5 = loader_docx2.load()

heading("Before Space Removal Pages5")

print(pages5)

#remove_extra_spaces_from_pages(pages5)

heading("After Space Removal Pages5")

print(pages5)


md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = [("#", "Course Title"), 
                                                                ("##", "Lecture Title")])

pages_md_split = md_splitter.split_text(pages5[0].page_content)

heading("pages_md_split")
print(pages_md_split)

for p in pages_md_split:
    heading2("Markdown Split Page", "")
    heading2("Page Metadata", p.metadata)
    heading2("Page Content", p.page_content)

# Need to study Token Text Splitters
# Need to study Recursive Character Text Splitters
