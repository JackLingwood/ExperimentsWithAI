# https://www.pinecone.io
# Setup PineCone Basic Vector Database 2
# ---------------------------------------------------------------------------------------------
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
heading(f"{name} PineCone Vector Database - Basic Setup")
api_key = os.environ.get("api_key")
pinecone_vector_database_key = os.environ.get("pinecone_vector_database_key")
pinecone_environment = os.environ.get("pinecone_environment", "gcp-starter")
# ---------------------------------------------------------------------------------------------

import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override = True)

pc = Pinecone(api_key = pinecone_vector_database_key, environment = pinecone_environment)

heading2("pc.list_indexes()", pc.list_indexes())

# pc.list_indexes()

index_name = "my-index"
dimension = 3
metric = "cosine"

heading2("Creating index", index_name)
heading3("Index Name", index_name)
heading3("Dimension", dimension)
heading3("Metric", metric)

heading2("pc.list_indexes()", pc.list_indexes())

def delete_index_if_exists(index_name):
    """
    Deletes the index if it exists.
    """
    note(f"Deleting index if it exists...{index_name}")
    if index_name in [index.name for index in pc.list_indexes()]:
        pc.delete_index(index_name)
        print(f"{index_name} succesfully deleted.")
    else:
        print(f"{index_name} not in index list.")

def create_index_if_not_exists(index_name, dimension, metric):
    """
    Creates an index if it does not exist.
    """
    note(f"Creating index if not exists...{index_name}")

    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            ),
        )
        print(f"{index_name} created successfully.")
    else:
        print(f"{index_name} already exists.")
    return pc.Index(index_name)


delete_index_if_exists(index_name)

index = create_index_if_not_exists(index_name=index_name, dimension=dimension, metric=metric)

index.upsert(
    [
        ("Dog", [4.0, 0.0, 1.0]),
        ("Cat", [4.0, 0.0, 1.0]),
        ("Chicken", [2.0, 2.0, 1.0]),
        ("Mantis", [6.0, 2.0, 3.0]),
        ("Elephant", [4.0, 0.0, 1.0]),
    ]
)

note("Index created and data upserted.")

heading2("pc",pc)


index_name_2 = "my-index-2"
dimension_2 = 1536
metric_2 = "euclidean"

heading3("Index Name", index_name_2)
heading3("Dimension", dimension_2)
heading3("Metric", metric_2)

delete_index_if_exists(index_name_2)
create_index_if_not_exists(index_name=index_name_2, dimension=dimension_2, metric=metric_2)

heading2("pc.list_indexes()", pc.list_indexes())

index = pc.Index(name=index_name)

note("Adding metadata to index...")


