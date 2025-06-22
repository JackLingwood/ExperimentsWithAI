import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = [' Most shark attacks occur about 10 feet from the beach since that is where the people are',
        'the efficiency with which he paired the socks in the drawer was quite admirable',
        'carol drank the blood as if she were a vampire',
        'giving directions that the mountains are to the west only works when you can see them',
        'the sign said there was road work ahead so he decided to speed up',
        'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms']

tfidfvec = TfidfVectorizer()

tfidfvec_fit = tfidfvec.fit_transform(data)

tfidf_bag = pd.DataFrame(tfidfvec_fit.toarray(), columns = tfidfvec.get_feature_names_out())

print(tfidf_bag)

# TF-IDF (Term Frequency-Inverse Document Frequency) is another popular text vectorization technique that builds on the Bag of Words model.
# It not only considers the frequency of words in a document but also their importance across the entire corpus.
# The TF-IDF model assigns a weight to each word based on its frequency in a document and its rarity across the corpus.
# This helps to reduce the impact of common words that appear frequently in many documents, while giving more weight to rare words that may be more informative.
# The TF-IDF model is particularly useful for tasks like document classification, information retrieval, and text mining.
# The TfidfVectorizer class from the sklearn library is used to implement the TF-IDF model.
# The fit_transform method is called on the TfidfVectorizer instance to learn the vocabulary and transform the text data into a matrix of TF-IDF scores.
# The resulting DataFrame contains the TF-IDF scores for each document, with words as columns and documents as rows.
# The get_feature_names_out method is used to retrieve the names of the features (words) in the vocabulary.
# The output DataFrame can be used for further analysis or as input to machine learning models.
# The TF-IDF model is widely used in text classification tasks, such as sentiment analysis, topic modeling, and information retrieval.
# The TF-IDF model can be customized with various parameters, such as ngram_range to specify the size of n-grams, stop_words to remove common words, and max_features to limit the vocabulary size.
# The TF-IDF model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.
# The TF-IDF model can be extended with techniques like n-grams, which consider sequences of words (bigrams, trigrams, etc.) to capture some context.
# The TF-IDF model can be used in conjunction with machine learning algorithms like Naive Bayes, Logistic Regression, or Support Vector Machines for text classification tasks.
