# RAG Application has these parts
# 1. Indexing - preparing data and storing it in specialized databased
# 2. Retrieval: Retrieve relevant documents from a vector store.
# 3. Augmentation: Combine the retrieved documents with a user query.
# 4. Generation: Generate a response based on the augmented query.

# Indexing is not covered in this example, but it is a crucial step in RAG applications.
# 1. Load Data into standard LangChain document format
# 2. Split Data into smaller chunks that can fit into the context window of the LLM
# 3. Embed the chunks into vector representations
# 4. Store the vectors in a vector database (e.g., Pinecone, Weaviate, ChromaDB, etc.)

# Retrieval is the process of fetching relevant documents from a vector store based on a user query.
# 1. Take user query
# 2. Embed the query into a vector representation
# 3. Search the vector store for similar vectors
# 4. Return the retrieved documents

# Augmentation is the process of combining the retrieved documents with the user query to provide context for the LLM.
# 1. Take user query and retrieved documents
# 2. Combine the query and documents into a single input for the LLM
# 3. Format the input in a way that the LLM can understand (e.g., using a prompt template)
# 4. Pass the combined input to the LLM for response generation

# Generation is the process of generating a response based on the augmented query.
# 1. User query and retrieved documents are passed to the LLM
# 2. LLM generates a response based on the input
# 3. Response is returned to the user

# Disadvantages of RAG:
# 1. Complexity: RAG applications are more complex than traditional LLM applications due to the additional steps of retrieval and augmentation.
# 2. Latency: The retrieval step can introduce latency, especially if the vector store
#    is large or if the query is complex.
# 3. Dependency on Vector Store: The performance of the RAG application is heavily dependent
#    on the quality and efficiency of the vector store used for retrieval.
# 4. Maintenance: Keeping the vector store up-to-date with new data can be challenging,
#    especially in dynamic environments where data changes frequently.

# Indexing has 4 steps:
# 1. Load Data
# 2. Split Data
# 3. Embed Data
# 4. Store Data

# Load Data involveds
# Document loaders that can read various file formats (e.g., PDF, CSV, JSON, etc.)
# Produce single format documents that can be processed by LangChain.

# How load from PDF + DOCX
# Content must visit limit of context window.
# Too much data in documents leads to excessive token consumption.

# LLM does better with smaller chunks.
# Each chunk should center around one topic.

# Embedding data involves converting text into vector representations.
# Embeddings are numerical representations of text that capture semantic meaning.
# LangChain provides various embedding models that can be used to convert text into vectors.

# Each vector is a fixed-length array of numbers that represents the semantic meaning of the text.
# These vectors can be stored in a vector database for efficient retrieval.
# Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently.

# Each input gets a different vector.
# Similar inputs will have similar vectors.
# Vector databases use algorithms like cosine similarity or Euclidean distance to find similar vectors.
# LangChain provides various vector stores that can be used to store and retrieve vectors.


# Measuring the distance between two vectors is done using distance metrics such as
# Dot Product, Cosine Similarity, Euclidean Distance, etc.
# These metrics help determine how similar or dissimilar two vectors are.
# The most common distance metric is Cosine Similarity, which measures the cosine of the angle between two vectors.


# Cosine similarity is a measure of similarity between two non-zero vectors.
# It is defined as the cosine of the angle between the two vectors.
# Cosine similarity is calculated as the dot product of the two vectors divided by the product of their magnitudes.
# Cosine similarity ranges from -1 to 1, where 1 means the vectors are
# identical, 0 means they are orthogonal (no similarity), and -1 means they are diametrically opposed.

# The formula for calculating the dot product of two vectors is:
# dot_product = sum(a_i * b_i for i in range(n))
# where a_i and b_i are the components of the vectors a and b, respectively, and n is the number of dimensions.

# The formula for calculating the cosine similarity between two vectors is:
# cosine_similarity = dot_product(a, b) / (magnitude(a) * magnitude(b))
# where magnitude(a) is the square root of the sum of the squares of the components of vector a.

# The formula for calculating the magnitude of a vector is:
# magnitude(a) = sqrt(sum(a_i^2 for i in range(n)))

# The formula for calculating the Euclidean distance between two vectors is:
# euclidean_distance = sqrt(sum((a_i - b_i)^2 for i in range
# n))

# The formula for calculating the Manhattan distance between two vectors is:
# manhattan_distance = sum(abs(a_i - b_i) for i in range(n))

# The formula for calculating the Jaccard similarity between two sets is:
# jaccard_similarity = len(set_a.intersection(set_b)) / len(set_a.union(set_b))

# The formula for calculating the Jaccard distance between two sets is:
# jaccard_distance = 1 - jaccard_similarity

# The formula for calculating the Hamming distance between two strings is:
# hamming_distance = sum(1 for a_i, b_i in zip(string_a, string_b) if a_i != b_i)

# The formula for calculating the Levenshtein distance between two strings is:
# levenshtein_distance = min(
#     levenshtein_distance(string_a[:-1], string_b) + 1,
#     levenshtein_distance(string_a, string_b[:-1]) + 1,
#     levenshtein_distance(string_a[:-1], string_b[:-1]) + cost
# )
# where cost is 0 if the last characters are the same, and 1 if they are different.

# The formula for calculating the Pearson correlation coefficient between two vectors is:
# pearson_correlation = covariance(a, b) / (std_dev(a) * std
# dev(b))
# where covariance(a, b) is the covariance between vectors a and b, and std_dev(a) and std_dev(b) are the standard deviations of vectors a and b, respectively.

# The formula for calculating the Spearman rank correlation coefficient between two vectors is:
# spearman_rank_correlation = 1 - (6 * sum(d_i^2 for i in range(n))) / (n * (n^2 - 1))
# where d_i is the difference between the ranks of the i-th elements of the two vectors, and n is the number of elements in the vectors.

# The formula for calculating the Kendall tau coefficient between two vectors is:
# kendall_tau = (number of concordant pairs - number of discordant pairs) / (n * (n - 1) / 2)
# where n is the number of elements in the vectors, and a pair is concordant if the ranks of the elements in one vector are the same as the ranks of the elements in the other vector, and discordant if they are different.

# The formula for calculating the Minkowski distance between two vectors is:
# minkowski_distance = (sum(abs(a_i - b_i)^p for i in range(n)))^(1/p)
# where p is the order of the distance (e.g., p=1 for Manhattan distance, p=2 for Euclidean distance, etc.).

# The formula for calculating the Mahalanobis distance between two vectors is:
# mahalanobis_distance = sqrt((x - mu).T @ inv(S) @ (x - mu))
# where x is the vector, mu is the mean vector, S is the covariance matrix, and @ denotes matrix multiplication.

# The formula for calculating the Bray-Curtis dissimilarity between two vectors is:
# bray_curtis_dissimilarity = sum(abs(a_i - b_i) for
# i in range(n)) / sum(a_i + b_i for i in range(n))

# The formula for calculating the Canberra distance between two vectors is:
# canberra_distance = sum(abs(a_i - b_i) / (abs(a_i) + abs(b_i)) for i in range(n))

# The formula for calculating the Chebyshev distance between two vectors is:
# chebyshev_distance = max(abs(a_i - b_i) for i in range(n))

# The formula for calculating the Cosine distance between two vectors is:
# cosine_distance = 1 - (dot_product(a, b) / (magnitude(a) * magnitude(b)))

# The formula for calculating the Cosine distance between two vectors is:
# cosine_distance = 1 - (dot_product(a, b) / (magnitude(a) * magnitude(b)))


# Vectors are stored in a vector database.
# Relational databases are not suitable for storing vectors.
# Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently.
# Vector databases use algorithms like cosine similarity or Euclidean distance to find similar vectors.
# LangChain provides various vector stores that can be used to store and retrieve vectors.
# Vector stores are used to store and retrieve vectors efficiently.

# We do not need exact matches.
# We need to find semantically similar vectors.

# Retrievers are the part of the RAG chain responsible for retrieving the most relevant and diverse parts.
# Diversity is important to ensure that the retrieved documents cover different aspects of the query.
# Retrievers can be used to filter out irrelevant documents and ensure that the retrieved documents are relevant to the query.

# Augmentation is the process of combining the retrieved documents with the user query to provide context for the LLM.
# Augmentation can be done using prompt templates that format the input in a way that the L

# pip install pypdf
# pip install docx2txt







