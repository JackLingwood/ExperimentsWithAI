import nltk
import pandas as pd
import matplotlib.pyplot as plt

tokens = ['the', 'rise', 'of', 'artificial', 'intelligence', 'has', 'led', 'to', 'significant', 'advancements', 'in', 'natural', 'language', 'processing', 'computer', 'vision', 'and', 'other', 'fields', 'machine', 'learning', 'algorithms', 'are', 'becoming', 'more', 'sophisticated', 'enabling', 'computers', 'to', 'perform', 'complex', 'tasks', 'that', 'were', 'once', 'thought', 'to', 'be', 'the', 'exclusive', 'domain', 'of', 'humans', 'with', 'the', 'advent', 'of', 'deep', 'learning', 'neural', 'networks', 'have', 'become', 'even', 'more', 'powerful', 'capable', 'of', 'processing', 'vast', 'amounts', 'of', 'data', 'and', 'learning', 'from', 'it', 'in', 'ways', 'that', 'were', 'not', 'possible', 'before', 'as', 'a', 'result', 'ai', 'is', 'increasingly', 'being', 'used', 'in', 'a', 'wide', 'range', 'of', 'industries', 'from', 'healthcare', 'to', 'finance', 'to', 'transportation', 'and', 'its', 'impact', 'is', 'only', 'set', 'to', 'grow', 'in', 'the', 'years', 'to', 'come']
print(tokens)

# unigrams: n=1
unigrams = (pd.Series(nltk.ngrams(tokens, 1)).value_counts()) 
print(unigrams)

unigrams[0:10].sort_values().plot.barh(color='lightsalmon', width=.9, figsize=(12, 8))
plt.title('10 Most Frequently Occuring Unigrams')
plt.ylabel('Unigram')
plt.xlabel('# of Occurances')
plt.show()

# bigrams: n=2
bigrams = (pd.Series(nltk.ngrams(tokens, 2)).value_counts()) 
print(bigrams)

bigrams[0:10].sort_values().plot.barh(color='skyblue', width=.9, figsize=(12, 8))

plt.title('10 Most Frequently Occuring Bigrams')
plt.ylabel('Bigram')
plt.xlabel('# of Occurences')
plt.show()

# trigrams: n=3
trigrams = (pd.Series(nltk.ngrams(tokens, 3)).value_counts())
print(trigrams)