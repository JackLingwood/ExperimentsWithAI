# https://www.pinecone.io
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
heading(f"{name} PineCone Vector Database - Load FineWeb Dataset")
api_key = os.environ.get("api_key")
pinecone_vector_database_key = os.environ.get("pinecone_vector_database_key")
pinecone_environment = os.environ.get("pinecone_environment", "gcp-starter")
# ---------------------------------------------------------------------------------------------

from datasets import load_dataset
from dotenv import load_dotenv, find_dotenv
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer

fw = load_dataset("HuggingFaceFW/fineweb", name = "sample-10BT", split = "train", streaming = True)

heading2("fineweb dataset", fw)

heading2("fineweb features", fw.features)

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key = pinecone_vector_database_key, environment = pinecone_environment)

heading2("pc.list_indexes()", pc.list_indexes())

heading("Uploading a subset of FineWeb to Pinecone")
# We must use same number of dimensions as the model we are using

dimensionCount = model.get_sentence_embedding_dimension()
heading3("Dimension Count", dimensionCount)

pc.create_index(
    name="text",
    dimension=model.get_sentence_embedding_dimension(),
    metric="cosine",
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

index = pc.Index(name = "text")

# Define the number of items you want to process (subset size)
subset_size = 10000  # For example, take only 10,000 items

note(f"Processing a subset of size: {subset_size}")

# Iterate over the dataset and prepare data for upserting
vectors_to_upsert = []
for i, item in enumerate(fw):

    if (i%1000) == 0:
        note(f"Encoding item {i}")

    if i >= subset_size:
        break

    text = item['text']
    unique_id = str(item['id'])
    language = item['language']

    # Create an embedding for the text
    embedding = model.encode(text, show_progress_bar=False).tolist()

    # Prepare metadata
    metadata = {'language': language}

    # Append the tuple (id, embedding, metadata) to the list
    vectors_to_upsert.append((unique_id, embedding, metadata))

# Upsert data to Pinecone in batches
batch_size = 1000  # Adjust based on your environment and dataset size
for i in range(0, len(vectors_to_upsert), batch_size):
    note(f"Upserting batch {i // batch_size + 1}")
    batch = vectors_to_upsert[i:i + batch_size]
    index.upsert(vectors=batch)

note("Subset of data upserted to Pinecone index.")