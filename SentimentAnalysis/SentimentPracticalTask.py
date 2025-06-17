import sys
import os
sys.path.append(os.path.abspath("Shared"))  # Add the parent directory to the system path
from utils import heading, clearConsole, setCurrentDirectory
clearConsole()

setCurrentDirectory(__file__)

import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import transformers
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import chain
from nltk import NaiveBayesClassifier

data = pd.read_csv("book_reviews_sample.csv") # the data should be in the same folder as your notebook
print(data.head())
print(data.info())
print(data['reviewText'][0])

# lowercase
data['reviewText_clean'] = data['reviewText'].str.lower()

# remove punctuation
data['reviewText_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['reviewText_clean']), axis=1)

print(data.head())

vader_sentiment = SentimentIntensityAnalyzer()

data['vader_sentiment_score'] = data['reviewText_clean'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])

print(data['vader_sentiment_score'].head())


# create labels
bins = [-1, -0.1, 0.1, 1]
names = ['negative', 'neutral', 'positive']

data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

import matplotlib.pyplot as plt
data['vader_sentiment_label'].value_counts().plot.bar()
plt.show()

transformer_pipeline = pipeline("sentiment-analysis")

transformer_labels = []

for review in data['reviewText_clean'].values:
    sentiment_list = transformer_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)
    
data['transformer_sentiment_label'] = transformer_labels

data['transformer_sentiment_label'].value_counts().plot.bar()

plt.show()

print(data.head() )