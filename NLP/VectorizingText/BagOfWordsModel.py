import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

data = [' Most shark attacks occur about 10 feet from the beach since that is where the people are',
        'the efficiency with which he paired the socks in the drawer was quite admirable',
        'carol drank the blood as if she were a vampire',
        'giving directions that the mountains are to the west only works when you can see them',
        'the sign said there was road work ahead so he decided to speed up',
        'the gruff old man sat in the back of the bait shop grumbling to himself as he scooped out a handful of worms'
        ]

countvec = CountVectorizer()

countvec_fit = countvec.fit_transform(data)

bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())

print(bag_of_words)


# Text vectorization is the process of converting text into a numerical representation that can be used by machine learning algorithms.
# Bag of Words (BoW) is a simple and commonly used text vectorization technique that represents text data as a collection of words, disregarding grammar and word order.
# It creates a matrix where each row represents a document and each column represents a unique word from the entire corpus.
# The value in each cell of the matrix indicates the frequency of the corresponding word in the document.
# The CountVectorizer class from the sklearn library is used to implement the Bag of Words model.
# The fit_transform method is called on the CountVectorizer instance to learn the vocabulary and transform the text data into a matrix of token counts.
# The resulting DataFrame contains the word counts for each document, with words as columns and documents as rows.
# The get_feature_names_out method is used to retrieve the names of the features (words) in the vocabulary.
# The output DataFrame can be used for further analysis or as input to machine learning models.
# The Bag of Words model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.
# Note: The Bag of Words model does not consider the order of words or their context, which can lead to loss of information in some cases.
# The Bag of Words model is simple and effective for many text classification tasks, but it has limitations such as ignoring word order and context.
# It is often used as a baseline for more advanced text vectorization techniques like TF-IDF or word embeddings.
# The Bag of Words model is widely used in text classification tasks, such as sentiment analysis, topic modeling, and spam detection.
# The Bag of Words model can be extended with techniques like n-grams, which consider sequences of words (bigrams, trigrams, etc.) to capture some context.
# The Bag of Words model can be used in conjunction with machine learning algorithms like Naive Bayes, Logistic Regression, or Support Vector Machines for text classification tasks.
# The Bag of Words model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.

# TF-IDF (Term Frequency-Inverse Document Frequency) is another popular text vectorization technique that builds on the Bag of Words model.
# It not only considers the frequency of words in a document but also their importance across the entire corpus.
# The TF-IDF model assigns a weight to each word based on its frequency in a document and its rarity across the corpus.
# This helps to reduce the impact of common words that appear frequently in many documents, while giving more weight to rare words that may be more informative.
# The TF-IDF model is particularly useful for tasks like document classification, information retrieval, and text mining.
# The CountVectorizer class can be customized with various parameters, such as ngram_range to specify the size of n-grams, stop_words to remove common words, and max_features to limit the vocabulary size.
# The Bag of Words model can be used in various applications, including text classification, sentiment analysis, topic modeling, and information retrieval.
# The Bag of Words model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.
# The Bag of Words model can be extended with techniques like n-grams, which consider sequences of words (bigrams, trigrams, etc.) to capture some context.
# The Bag of Words model can be used in conjunction with machine learning algorithms like Naive Bayes, Logistic Regression, or Support Vector Machines for text classification tasks.
# The Bag of Words model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.
# The Bag of Words model can be extended with techniques like n-grams, which consider sequences of words (bigrams, trigrams, etc.) to capture some context.
# The Bag of Words model can be used in conjunction with machine learning algorithms like Naive Bayes, Logistic Regression, or Support Vector Machines for text classification tasks.
# The Bag of Words model is a foundational technique in natural language processing and is often used as a baseline for text classification tasks.
# The Bag of Words model can be extended with techniques like n-grams, which consider sequences of words (bigrams, trigrams, etc.) to capture some context.


