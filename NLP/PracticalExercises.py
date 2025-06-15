import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
import pandas as pd

import os


def clear_console():
       os.system('cls' if os.name == 'nt' else 'clear')



print("Current working directory:", os.getcwd())
print("File path:", __file__, "File name:", os.path.basename(__file__))
print("File directory:", os.path.dirname(__file__))

os.chdir(os.path.dirname(__file__))
print("Current working directory:", os.getcwd())

# Load the dataset
data = pd.read_csv("tripadvisor_hotel_reviews.csv")
                    
# Ensure necessary NLTK resources are downloaded

data.info()
clear_console()

def reviewdata(data):
    print("First 5 rows of the dataset:")
    print(data.head())
    print("\nColumns in the dataset:")
    print(data.columns)
    print("\nData types of the columns:")
    print(data.dtypes)
    print("\nMissing values in the dataset:")
    print(data.isnull().sum())
    print("\nDescriptive statistics of the dataset:")
    print(data.describe())
    print("\nFirst 5 rows of the 'Review' column:")
    print(data['Review'].head())
    # Check if the 'Review' column exists
    if 'Review' not in data.columns:
        raise ValueError("The 'Review' column is not present in the dataset.")
    # Check if the 'Review' column is empty
    if data['Review'].isnull().all():
        raise ValueError("The 'Review' column is empty.")
    # Check if the 'Review' column contains any non-string values
    if not all(isinstance(x, str) for x in data['Review']):
        raise ValueError("The 'Review' column contains non-string values.")
    # Check if the 'Review' column contains any empty strings
    if data['Review'].str.strip().eq('').any():
        raise ValueError("The 'Review' column contains empty strings.")
    # Check if the 'Review' column contains any NaN values
    if data['Review'].isnull().any():
        raise ValueError("The 'Review' column contains NaN values.")
    # Check if the 'Review' column contains any duplicate values
    if data['Review'].duplicated().any():
        raise ValueError("The 'Review' column contains duplicate values.")
    # Check if the 'Review' column contains any special characters
    if data['Review'].str.contains(r'[^a-zA-Z0-9\s]', regex=True).any():
        print ("The 'Review' column contains special characters.")
    # Check if the 'Review' column contains any numbers
    if data['Review'].str.contains(r'\d', regex=True).any():
        print ("The 'Review' column contains numbers.")
    # Check if the 'Review' column contains any URLs
    if data['Review'].str.contains(r'http[s]?://', regex=True).any():
        raise ValueError("The 'Review' column contains URLs.")
    # Check if the 'Review' column contains any email addresses
    if data['Review'].str.contains(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', regex=True).any():
        raise ValueError("The 'Review' column contains email addresses.")


print("First review in the dataset:")
print(data['Review'][0])
data['review_lowercase'] = data['Review'].str.lower()
print(data.head())

en_stopwords = stopwords.words('english')
en_stopwords.remove('not')  # Remove 'not' from stopwords

print(data.info())
print(data.head())
print(data['Review'][0])

print ("\nREMOVING STOPWORDS")

data['review_no_stopwords'] = data['review_lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))




print(data.head())
print(data['review_no_stopwords'][0])

print ("\nREMOVING PUNCTUATION")

data['review_no_stopwords_no_punct'] = data \
                                    .apply(lambda x: \
                                    re.sub(r"[*]", \
                                    "star", \
                                    x['review_no_stopwords'] \
                                    ), axis=1)
print(data.head())

data['review_no_stopwords_no_punct'] = data. \
                                    apply(lambda x: \
                                    re.sub(r"([^\w\s])", \
                                    "", \
                                    x['review_no_stopwords_no_punct'] \
                                    ), axis=1)
print(data.head())

print ("\nTOKENIZATION")

data['tokenized'] = data.apply(lambda x: \
                               word_tokenize( \
                               x['review_no_stopwords_no_punct'] \
                               ), axis=1)
print(data.head())
print(data['tokenized'][0])

print ("\nSTEMMING AND LEMMATIZATION")

ps = PorterStemmer()
data["stemmed"] = data["tokenized"] \
                  .apply(lambda tokens: \
                  [ps.stem(token) \
                   for token in tokens])
print(data.head())
print(data['stemmed'][0])

lemmatizer = WordNetLemmatizer()
data["lemmatized"] = data["tokenized"] \
                    .apply(lambda tokens: \
                    [lemmatizer.lemmatize(token) \
                     for token in tokens])
print(data['lemmatized'][0])
print(data.head())

tokens_clean = sum(data['lemmatized'], [])

print ("\nUNIGRAMS")

# unigrams: n=1
unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()) 
print(unigrams)

print ("\nBIGRAMS")

# bigrams: n=2
bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()) 
print(bigrams)

# trigrams: n=3
print ("\nTRIGRAMS")
ngrams_4 = (pd.Series(nltk.ngrams(tokens_clean, 3)).value_counts()) 
print(ngrams_4)