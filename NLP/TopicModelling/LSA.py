# LSA Model
# Latent Semantic Analysis (LSA) is a technique in natural language processing and information retrieval that helps to identify patterns in the relationships between terms and concepts in a text corpus.
# It is often used for topic modeling, document clustering, and dimensionality reduction.
# LSA works by decomposing a term-document matrix into a lower-dimensional representation, capturing the underlying structure of the data.
# It uses singular value decomposition (SVD) to reduce the dimensionality of the term-document matrix, allowing for the identification of latent semantic structures.

# 1. Distributional Hypothesis
# The distributional hypothesis states that words that occur in similar contexts tend to have similar meanings.
# 2. Singular Value Decomposition (SVD)
# SVD is a mathematical technique used to decompose a matrix into three matrices, capturing the underlying structure of the data.
# SVD recreates text documents into different vectors, allowing for the identification of latent semantic structures.
# Each vector expresses a different way of looking at meaning in the text.
# M is the term-document matrix, U is the left singular vectors, S is the singular values, and V is the right singular vectors.
# M is the document-term matrix, where each row represents a document and each column represents a term.
# The value in each cell of the matrix represents the frequency of the corresponding term in the document.
# U is a document topic matrix, where each row represents a document and each column represents a topic.
# S is vector where values are the singular values, which represent the importance of each topic.
# Vt is a term topic matrix, where each row represents a term and each column represents a topic.







# 3. Dimensionality Reduction
# LSA reduces the dimensionality of the term-document matrix, allowing for the identification of latent semantic structures.

import sys
import os
sys.path.append(os.path.abspath("Shared"))  # Add the parent directory to the system path
print(sys.path)
from utils import heading, clearConsole
clearConsole()

os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

heading("Topic Modelling with LSA")



import pandas as pd

import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora

from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


data = pd.read_csv("news_articles.csv")
heading("New Articles Data")

print(data.head())
print(data.info())



# take just the content of the article, lowercase and remove punctuation
articles = data['content'].str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))

# stop word removal
en_stopwords = stopwords.words('english')
articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# tokenize
articles = articles.apply(lambda x: word_tokenize(x))

heading("Cleaned up Articles")
print(articles)


heading("Stemmed Articles")
# stemming (done for speed as we have a lot of text)
ps = PorterStemmer()
articles = articles.apply(lambda tokens: [ps.stem(token) for token in tokens])

print(articles)

heading("Dictionary and Document Term Matrix")

# create dictionary of all words
dictionary = corpora.Dictionary(articles)
print(dictionary)

# vecotize using bag of words into a document term matrix
doc_term = [dictionary.doc2bow(text) for text in articles]

# heading("Document Term Matrix")
print(doc_term)


# specify number of topics
num_topics = 2

heading("LSA Model")

# create LSA model
lsamodel = LsiModel(doc_term, num_topics=num_topics, id2word = dictionary) 
print(lsamodel.print_topics(num_topics=num_topics, num_words=5))




# generate coherence scores to determine an optimum number of topics
coherence_values = []
model_list = []

min_topics = 2
max_topics = 11



def main():
    for num_topics_i in range(min_topics, max_topics+1):
        model = LsiModel(doc_term, num_topics=num_topics_i, id2word = dictionary, random_seed=0)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=articles, dictionary=dictionary, coherence='c_v')
        if __name__ == '__main__':
            coherence_values.append(coherence_model.get_coherence())

print("BEFORE LOOP")
main()

print("AFTER LOOP")

exit()

heading("Coherence Scores for Different Number of Topics")
print(coherence_values)

exit()
plt.plot(range(min_topics, max_topics+1), coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

final_n_topics = 3
lsamodel_f = LsiModel(doc_term, num_topics=final_n_topics, id2word = dictionary) 
print(lsamodel_f.print_topics(num_topics=final_n_topics, num_words=5))




