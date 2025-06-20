# Latent Dirichlet Allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. It is commonly used for topic modeling in text data.
# LDA assumes that documents are mixtures of topics, where each topic is a distribution over words. It infers the topics from the documents and assigns probabilities to each word in the vocabulary for each topic.
# LDA is particularly useful for discovering hidden thematic structure in large collections of documents.

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora

import sys
import os
sys.path.append(os.path.abspath("Shared"))  # Add the parent directory to the system path
print(sys.path)
from utils import heading, clearConsole
clearConsole()

os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

heading("Toic Modelling with LDA")




# LDA (Latent Dirichlet Allocation) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar.


data = pd.read_csv("news_articles.csv") # data should be in the same folder as your notebook

print(data.head())

print(data.info())




# take just the content of the article, lowercase and remove punctuation
articles = data['content'].str.lower().apply(lambda x: re.sub(r"([^\w\s])", "", x))

# stop word removal
en_stopwords = stopwords.words('english')
articles = articles.apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

# tokenize
articles = articles.apply(lambda x: word_tokenize(x))

# stemming (done for speed as we have a lot of text)
ps = PorterStemmer()
articles = articles.apply(lambda tokens: [ps.stem(token) for token in tokens])

heading("Articles after pre-processing")
print(articles)

# create dictionary of all words
dictionary = corpora.Dictionary(articles)

heading("Dictionary of words")
print(dictionary)

heading("Document Term Matrix")

# vecotize using bag of words into a document term matrix
doc_term = [dictionary.doc2bow(text) for text in articles]

print(doc_term)

heading("Topics")

# specify number of topics
num_topics = 2

# create LDA model
lda_model = gensim.models.LdaModel(corpus=doc_term,
                                   id2word=dictionary,
                                   num_topics=num_topics)


print(lda_model.print_topics(num_topics=num_topics, num_words=5))