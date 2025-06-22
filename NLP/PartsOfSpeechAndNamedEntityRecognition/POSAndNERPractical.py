import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath("Shared"))  # Add the parent directory to the system path
print(sys.path)
from utils import heading, clearConsole
clearConsole()

heading("Parts of Speech and Named Entity Recognition Practical")

def ChangeToCurrentFileDirectory():
    import os
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    # Change to the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    print("Changed working directory to script location:", script_directory)

ChangeToCurrentFileDirectory()

bbc_data = pd.read_csv("bbc_news.csv") # the data should be in the same folder as your notebook

heading("BBC Data")
print(bbc_data)

heading("BBC Data Head")
print(bbc_data.head())

heading("BBC Data Info")
print(bbc_data.info())


heading("BBC Titles Dataframe")
titles = pd.DataFrame(bbc_data['title'])
print(titles)

heading("BBC Titles DataFrame Head")
print(titles.head())




# lowercase
titles['lowercase'] = titles['title'].str.lower()

heading("BBC Titles DataFrame Head - After lowercasing")
print(titles.head())



# stop word removal
en_stopwords = stopwords.words('english')
titles['no_stopwords'] = titles['lowercase'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

heading("BBC Titles DataFrame Head - After stop word removal")
print(titles.head())

# punctation removal
titles['no_stopwords_no_punct'] = titles.apply(lambda x: re.sub(r"([^\w\s])", "", x['no_stopwords']), axis=1)
heading("BBC Titles DataFrame Head - After punctuation removal")
print(titles.head())


# tokenize
titles['tokens_raw'] = titles.apply(lambda x: word_tokenize(x['title']), axis=1)
titles['tokens_clean'] = titles.apply(lambda x: word_tokenize(x['no_stopwords_no_punct']), axis=1)
heading("BBC Titles DataFrame Head - After tokenization")
print(titles.head())

# lemmatizing 
lemmatizer = WordNetLemmatizer()
titles["tokens_clean_lemmatized"] = titles["tokens_clean"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
heading("BBC Titles DataFrame Head - After lemmatization")
print(titles.head())

# create lists for just our tokens
tokens_raw_list = sum(titles['tokens_raw'], []) #unpack our lists into a single list
heading("BBC Tokens Raw List")
print(tokens_raw_list)



tokens_clean_list = sum(titles['tokens_clean_lemmatized'], [])

heading("BBC Tokens Clean List")
print(tokens_clean_list)

nlp = spacy.load('en_core_web_sm')

# create a spacy doc from our raw text - better for pos tagging
spacy_doc = nlp(' '.join(tokens_raw_list))

# extract the tokens and pos tags into a dataframe
pos_df = pd.DataFrame(columns=['token', 'pos_tag'])

for token in spacy_doc:
    pos_df = pd.concat([pos_df,
                       pd.DataFrame.from_records([{'token': token.text,'pos_tag': token.pos_}])], ignore_index=True)

# token frequency count
pos_df_counts = pos_df.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

heading("POS Tags Frequency Count")
print(pos_df_counts.head(10))


# most common nouns
nouns = pos_df_counts[pos_df_counts.pos_tag == "NOUN"][0:10]

heading("Most Common Nouns")
print(nouns)

# most common verbs
verbs = pos_df_counts[pos_df_counts.pos_tag == "VERB"][0:10]

heading("Most Common Verbs")
print(verbs)

# most common adjectives
adj = pos_df_counts[pos_df_counts.pos_tag == "ADJ"][0:10]
heading("Most Common Adjectives")
print(adj)

# extract the tokens and entity tags into a dataframe
ner_df = pd.DataFrame(columns=['token', 'ner_tag'])

for token in spacy_doc.ents:
    if pd.isna(token.label_) is False:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records(
            [{'token': token.text, 'ner_tag': token.label_}])], ignore_index=True)

heading("Named Entity Recognition DataFrame")
print(ner_df.head())


# token frequency count
ner_df_counts = ner_df.groupby(['token','ner_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)

heading("NER Tags Frequency Count")
print(ner_df_counts.head(10))


# most common people
people = ner_df_counts[ner_df_counts.ner_tag == "PERSON"][0:10]
heading("Most Common People")
print(people)


# most common places
places = ner_df_counts[ner_df_counts.ner_tag == "GPE"][0:10]
heading("Most Common Places")
print(places)
places