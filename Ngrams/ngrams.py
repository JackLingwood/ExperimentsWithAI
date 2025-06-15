# Text ban be broken into n-grams to check pre-processing.
# N-Gram is sequence of neighboring n words or tokens, n can be any number
import nltk
import pandas as pd
import matplotlib.pyplot as plt


print ("All of the token in a single string")
print("="*40)
# nltk.download('punkt') # Uncomment if you need to download the punkt tokenizer
tokens = ['the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in', 'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning', 'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex', 'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with', 'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful', 'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in', 'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly', 'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance', 'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years', 'to', 'come']

# combine tokens into a single string for display
print(' '.join(tokens))
print("Total number of tokens:", len(tokens))
print("Total number of unique tokens:", len(set(tokens)))


unigrams = (pd.Series(nltk.ngrams(tokens,1))).value_counts() # Unigram = 1-gram n-gram 
print(unigrams[:10])

unigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12,8))
plt.title("10 Most frequently occurring unigrams")
plt.show()

bigrams = (pd.Series(nltk.ngrams(tokens,2))).value_counts() # Unigram = 1-gram n-gram 
print(bigrams[:10])

bigrams[:10].sort_values().plot.barh(color="lightsalmon", width=.9, figsize=(12,8))
plt.title("10 Most frequently occurring bigrams")
plt.show()

trigrams = (pd.Series(nltk.ngrams(tokens,3))).value_counts() # Unigram = 1-gram n-gram 
print(trigrams[:10])

# n-gram = anytime n > 3
# tripadvisor_hotel_reviews.csv