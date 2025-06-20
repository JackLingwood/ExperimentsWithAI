import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

sys.path.append(os.path.abspath("Shared"))
from utils import heading, clearConsole

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return tokens

def run_lsa():
    clearConsole()

    os.chdir(os.path.dirname(__file__))
    print("Current working directory:", os.getcwd())

    data = pd.read_csv("news_articles.csv")
    heading("News Articles Data")
    print(data.head())
    print(data.info())

    heading("Preprocessing Articles")
    data['processed'] = data['content'].apply(preprocess_text)
    print(data['processed'].head())

    heading("Dictionary and Document-Term Matrix")
    dictionary = corpora.Dictionary(data['processed'])
    doc_term_matrix = [dictionary.doc2bow(text) for text in data['processed']]
    print(dictionary)

    num_topics = 2
    heading(f"LSA Model with {num_topics} Topics")
    lsa_model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)
    print(lsa_model.print_topics(num_topics=num_topics, num_words=5))

    heading("Optimizing Number of Topics")
    min_topics, max_topics = 2, 8
    coherence_values, model_list = [], []

    for n_topics in range(min_topics, max_topics + 1):
        model = LsiModel(doc_term_matrix, num_topics=n_topics, id2word=dictionary)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=data['processed'], dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    plt.plot(range(min_topics, max_topics + 1), coherence_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores by Number of Topics")
    plt.grid()
    plt.show()

    optimal_topics = coherence_values.index(max(coherence_values)) + min_topics
    heading(f"Final LSA Model with {optimal_topics} Topics")
    final_lsa_model = LsiModel(doc_term_matrix, num_topics=optimal_topics, id2word=dictionary)
    print(final_lsa_model.print_topics(num_topics=optimal_topics, num_words=5))

if __name__ == '__main__': # This ensures the script runs only when executed directly, not when imported as a module
    run_lsa()
