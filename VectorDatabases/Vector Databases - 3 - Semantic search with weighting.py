# Embedding text and upserting to Pinecone Vector Database
# https://www.pinecone.io
# https://sbert.net/#
# Setup PineCone Basic Vector Database 2
# ---------------------------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
import os
import sys

# Add Shared folder to sys.path
sys.path.append(os.path.abspath("Shared"))

from utils import heading, heading2, heading3, clearConsole, note
from openai import OpenAI

# Load environment variables
load_dotenv(find_dotenv(), override = True)

# Clear console and set working directory
clearConsole()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print("Current working directory:", os.getcwd())

# Headings
name = os.path.basename(__file__)
heading(f"{name}")
api_key = os.environ.get("api_key")
pinecone_vector_database_key = os.environ.get("pinecone_vector_database_key")
pinecone_environment = os.environ.get("pinecone_environment", "gcp-starter")
# ---------------------------------------------------------------------------------------------

### Indexing: Creating a Chroma Vectorstore


import pandas as pd

files = pd.read_csv("course_section_descriptions.csv", encoding='ANSI')

## Data preprocessing

from sentence_transformers import SentenceTransformer
import numpy as np

# Assuming 'files' DataFrame is already defined and includes necessary columns

# Define weights for different text components
weight_course_name = 5
weight_section_name = 3
weight_section_description = 2
weight_other = 1  # For other components like course technology and course description

# Create a unique identifier for each section combining course_id and section_id
files['unique_id'] = files['course_id'].astype(str) + '-' + files['section_id'].astype(str)

# Create metadata for each section
files['metadata'] = files.apply(lambda row: {
    "course_name": row['course_name'],
    "section_name": row['section_name'],
    "section_description": row['section_description']
}, axis=1)

def create_embeddings(row):
    # Encode individual components
    emb_course_name = model.encode(row['course_name'], show_progress_bar=False) * weight_course_name
    emb_section_name = model.encode(row['section_name'], show_progress_bar=False) * weight_section_name
    emb_section_description = model.encode(row['section_description'], show_progress_bar=False) * weight_section_description
    emb_course_tech = model.encode(row['course_technology'], show_progress_bar=False) * weight_other
    emb_course_desc = model.encode(row['course_description'], show_progress_bar=False) * weight_other

    # Combine embeddings by averaging them
    combined_embedding = (emb_course_name + emb_section_name + emb_section_description + emb_course_tech + emb_course_desc) / (weight_course_name + weight_section_name + weight_section_description + 2 * weight_other)
    return combined_embedding

# Initialize the model
model = SentenceTransformer('multi-qa-distilbert-cos-v1')

# Apply the function to create weighted embeddings
files['embedding'] = files.apply(create_embeddings, axis=1)

## Connect to Pinecone Index

import os
from pinecone import Pinecone, ServerlessSpec



import pinecone
pc = Pinecone(api_key = pinecone_vector_database_key, environment = pinecone_environment)

index = pc.Index("my-index")

index.delete(delete_all = True)

# Prepare the vectors for upserting
vectors_to_upsert = [(row['unique_id'], row['embedding'].tolist(), row['metadata']) for index, row in files.iterrows()  ]

# Upsert data
index.upsert(vectors=vectors_to_upsert)

print("Data successfully upserted to Pinecone index.")

import time
note("Waiting 20 seconds for Pinecone index to be ready...")
time.sleep(20)  # Waits for 20 seconds
note("Pinecone index is ready")


# Ensure you've already initialized and configured Pinecone and the model
# If not, you need to run the initialization code provided earlier

# Create the query embedding
query = "lasso"
query_embedding = model.encode(query, show_progress_bar=False).tolist()

query_results = index.query(
   # namespace="my-index",
    vector=[query_embedding],
    top_k=15,
    include_metadata=True
)

import pprint

score_threshold = 0.3

# Assuming query_results are fetched and include metadata
for match in query_results['matches']:
    if match['score'] >= score_threshold:
        course_details = match.get('metadata', {})
        course_name = course_details.get('course_name', 'N/A')
        section_name = course_details.get('section_name', 'N/A')
        section_description = course_details.get('section_description', 'No description available')
        
        pprint.pprint(f"Matched item ID: {match['id']}, Score: {match['score']}")
        pprint.pprint(f"Course: {course_name}")
        pprint.pprint(f"Section: {section_name}, Description: {section_description}", width = 100) 
       #pprint.pprint() 