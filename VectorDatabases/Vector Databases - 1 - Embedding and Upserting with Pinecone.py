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

files = pd.read_csv("course_descriptions.csv", encoding = "ANSI")

def create_course_description(row):
    return f'''The course name is {row["course_name"]}, the slug is {row["course_slug"]},
            the technology is {row["course_technology"]} and the course topic is {row["course_topic"]}'''

pd.set_option('display.max_rows', 106)
files['course_description_new'] = files.apply(create_course_description, axis = 1)

heading2("First 10 Course Descriptions", files["course_description"][:10])

pc = Pinecone(api_key = pinecone_vector_database_key, environment = pinecone_environment)

dimensionCount1 = 384
dimensionCount2 = 768

index_name = "my-index"
dimension = dimensionCount2
metric = "cosine"

modelName1 = "all-MiniLM-L6-v2" # dimension 384
modelName2 = "multi-qa-distilbert-cos-v1" # dimension 768
modelName = modelName2

def delete_index_if_exists(index_name):
    """
    Deletes the index if it exists.
    """
    note(f"Deleting index if it exists...{index_name}")
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)
        print(f"{index_name} successfully deleted.")
    else:
        print(f"{index_name} not in index list.")

delete_index_if_exists(index_name)

def create_index_if_not_exists(index_name, dimension, metric):
    note(f"Creating index if it does not exist...{index_name}")
    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name = index_name, 
            dimension = dimension, 
            metric = metric, 
            spec = ServerlessSpec(
                cloud = "aws", 
                region = "us-east-1")
            )
        print(f"{index_name} successfully created.")
    else:
        print(f"{index_name} already exists.")

create_index_if_not_exists(index_name, dimension, metric)

index = pc.Index(index_name)

heading2("Loadiing SentenceTransformer for model", modelName)
   
model = SentenceTransformer(modelName)

note("Model loaded successfully")

def create_embeddings(row):
    combined_text = ' '.join([str(row[field]) for field in ['course_description', 'course_description_new', 'course_description_short']])
    embedding = model.encode(combined_text, show_progress_bar = False)
    return embedding

files["embedding"] = files.apply(create_embeddings, axis = 1)
heading2("First 5 Embeddings", files["embedding"][:5])

vectors_to_upsert = [(str(row["course_name"]), row["embedding"].tolist()) for _, row in files.iterrows()]

heading3("Number of vectors to upsert", len(vectors_to_upsert))
heading2("First vector to upsert", vectors_to_upsert[:1])  # Display first vector for brevity

index.upsert(vectors = vectors_to_upsert)
note("Data upserted to Pinecone index")

import time

note("Waiting 20 seconds for Pinecone index to be ready...")
time.sleep(20)  # Waits for 20 seconds
note("Pinecone index is ready")

# Querying the Pinecone Index
query = "clustering"
heading2("Querying Pinecone Index with", query)

query_embedding = model.encode(query, show_progress_bar = False).tolist()

query_results = index.query(
    vector = [query_embedding],
    top_k = 12,
    include_values = True
)

heading2("Query Results", query_results)
heading2("Matched Items")

score_threshold = 0.1
for match in query_results["matches"]:
    if match['score'] >= score_threshold:
        print(f"Matched item ID: {match['id']}, score: {match['score']}")