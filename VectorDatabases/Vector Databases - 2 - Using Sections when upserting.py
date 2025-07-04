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

import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv
import pinecone
from sentence_transformers import SentenceTransformer

files = pd.read_csv("course_section_descriptions.csv", encoding = "ANSI")

files["unique_id"] = files["course_id"].astype(str) + '-' + files["section_id"].astype(str)

files["metadata"] = files.apply(lambda row: {
    "course_name": row["course_name"],
    "section_name": row["section_name"],
    "section_description": row["section_description"],
}, axis = 1)

def create_embeddings(row):
    combined_text = f'''{row["course_name"]} {row["course_technology"]}
                        {row["course_description"]} {row["section_name"]}{row["section_description"]}'''
    return model.encode(combined_text, show_progress_bar = False)

model = SentenceTransformer("multi-qa-distilbert-cos-v1")

files["embedding"] = files.apply(create_embeddings, axis = 1)

# Upserting data to Pinecone
load_dotenv(find_dotenv(), override = True)

pc = Pinecone(api_key = pinecone_vector_database_key, environment = pinecone_environment)

index_name = "my-index-3"
dimension = 768
metric = "cosine"

if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"{index_name} succesfully deleted.")
else:
     print(f"{index_name} not in index list.")

pc.create_index(
    name = index_name, 
    dimension = dimension, 
    metric = metric, 
    spec = ServerlessSpec(
        cloud = "aws", 
        region = "us-east-1")
    )

index = pc.Index(index_name)

vectors_to_upsert = [(row["unique_id"], row["embedding"].tolist(), row["metadata"]) for index, row in files.iterrows()  ]

index.upsert(vectors = vectors_to_upsert)
print("Data succesfully upserted to Pinecone index")

import time
note("Waiting 20 seconds for Pinecone index to be ready...")
time.sleep(20)  # Waits for 20 seconds
note("Pinecone index is ready")

query = "regression in Python"
query_embedding = model.encode(query, show_progress_bar=False).tolist()

query_results = index.query(
    vector = [query_embedding],
    top_k = 12,
    include_metadata=True
)

score_threshold = 0.4

heading2("Querying Pinecone Index with", query)
heading2("Matched Items")

# Assuming query_results are fetched and include metadata
for match in query_results['matches']:
    if match['score'] >= score_threshold:
        course_details = match.get('metadata', {})
        course_name = course_details.get('course_name', 'N/A')
        section_name = course_details.get('section_name', 'N/A')
        section_description = course_details.get('section_description', 'No description available')
        
        print(f"Matched item ID: {match['id']}, Score: {match['score']}")
        print(f"Course: {course_name} \nSection: {section_name} \nDescription: {section_description}")