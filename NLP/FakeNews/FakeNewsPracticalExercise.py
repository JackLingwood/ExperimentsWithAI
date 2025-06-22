import os
import sys

sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole

def main():
    clearConsole()

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    # Set current working directory to the script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)


    clearConsole()

    import pandas as pd
    import matplotlib.pyplot as plt

    # set plot options
    plt.rcParams['figure.figsize'] = (12, 8)
    default_plot_colour = "#00bfbf"

    heading("Fake News Classification")

    data = pd.read_csv("fake_news_data.csv")

    print("Number of articles in dataset:", len(data))
    print("Number of unique articles in dataset:", data['text'].nunique())

    heading2("data.head", data.head(10))
    heading2("data.info", data.info())


    # plot number of fake and factual articles
    data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_colour)
    plt.title('Count of Article Classification')
    plt.ylabel('# of Articles')
    plt.xlabel('Classification')
    plt.show()

    # POS = Part of Speech
    heading("Part of Speech (POS) Tagging and Named Entity Recognition (NER)")

    import seaborn as sns   
    import spacy
    from spacy import displacy
    from spacy import tokenizer
    import re
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import gensim
    import gensim.corpora as corpora
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.models import LsiModel, TfidfModel
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # spacy is a library for advanced NLP tasks, including POS tagging and NER
    # nltk is a library for basic NLP tasks, including tokenization and lemmatization
    # nltk requires some additional resources to be downloaded, uncomment the following lines to download them
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    # nlp.pipe is a method that allows us to process large amounts of text data efficiently

    nlp = spacy.load('en_core_web_sm')

    # split data by fake and factual news
    fake_news = data[data['fake_or_factual'] == "Fake News"]
    fact_news = data[data['fake_or_factual'] == "Factual News"]

    # create spacey documents - use pipe for dataframe
    fake_spaceydocs = list(nlp.pipe(fake_news['text']))
    fact_spaceydocs = list(nlp.pipe(fact_news['text'])) 

    # create function to extract tags for each document in our data
    def extract_token_tags(doc:spacy.tokens.doc.Doc):
        # this function extracts the text, named entity recognition (NER) tag, and part of speech (POS) tag for each token in a spacy document
        # it returns a list of tuples containing the token text, NER tag, and POS tag
        # doc is a spacy document object
        return [(i.text, i.ent_type_, i.pos_) for i in doc]

    # tag fake dataset 
    fake_tagsdf = []
    # create a list to hold the dataframes for each document
    # we will then concatenate these dataframes into a single dataframe
    # we will use the extract_token_tags function to extract the tags for each document
    import pandas as pd

    columns = ["token", "ner_tag", "pos_tag"]

    for ix, doc in enumerate(fake_spaceydocs):
        # extract the tags for each document
        tags = extract_token_tags(doc)
        # convert the tags to a dataframe
        tags = pd.DataFrame(tags)
        # set the column names
        tags.columns = columns
        # append the dataframe to the list
        # we will use this list to create a single dataframe at the end
        fake_tagsdf.append(tags)
            
    fake_tagsdf = pd.concat(fake_tagsdf)   

    # tag factual dataset 
    fact_tagsdf = []

    for ix, doc in enumerate(fact_spaceydocs):
        tags = extract_token_tags(doc)
        tags = pd.DataFrame(tags)
        tags.columns = columns
        fact_tagsdf.append(tags)
            
    fact_tagsdf = pd.concat(fact_tagsdf)

    heading2("fake_tagsdf.head", fake_tagsdf.head(10))

    # token frequency count (fake)
    pos_counts_fake = fake_tagsdf.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    heading2("pos_counts_fake.head", pos_counts_fake.head(10))

    # token frequency count (fact)
    pos_counts_fact = fact_tagsdf.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts', ascending=False)
    heading2("pos_counts_fact.head", pos_counts_fact.head(10))    

    # frequencies of pos tags
     
    heading2("pos_counts_fake.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10)",
             pos_counts_fake.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10))

    # frequencies of pos tags in factual news
    heading2("pos_counts_fact.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10)",
                pos_counts_fact.groupby(['pos_tag'])['token'].count().sort_values(ascending=False).head(10))


    # dive into diferences in nouns
    heading2("Nouns in Fake News", pos_counts_fake[pos_counts_fake.pos_tag == "NOUN"][0:15])
    heading2("Nouns in Factual News", pos_counts_fact[pos_counts_fact.pos_tag == "NOUN"][0:15])

    heading2("Verbs in Fake News", pos_counts_fake[pos_counts_fake.pos_tag == "VERB"][0:15])
    heading2("Verbs in Factual News", pos_counts_fact[pos_counts_fact.pos_tag == "VERB"][0:15])

    # plot the most common POS tags in fake news


    # top entities in fake news
    top_entities_fake = fake_tagsdf[fake_tagsdf['ner_tag'] != ""] \
                        .groupby(['token','ner_tag']).size().reset_index(name='counts') \
                        .sort_values(by='counts', ascending=False)

    heading2("top_entities_fake.head", top_entities_fake.head(15))

    # top entities in fact news
    top_entities_fact = fact_tagsdf[fact_tagsdf['ner_tag'] != ""] \
                        .groupby(['token','ner_tag']).size().reset_index(name='counts') \
                        .sort_values(by='counts', ascending=False)

    heading2("top_entities_fact.head", top_entities_fact.head(15))

    heading("Most Common Entities in Fake and Factual News")

    # create custom palette to ensure plots are consistent
    ner_palette = {
        'ORG': sns.color_palette("Set2").as_hex()[0],
        'GPE': sns.color_palette("Set2").as_hex()[1],
        'NORP': sns.color_palette("Set2").as_hex()[2],
        'PERSON': sns.color_palette("Set2").as_hex()[3],
        'DATE': sns.color_palette("Set2").as_hex()[4],
        'CARDINAL': sns.color_palette("Set2").as_hex()[5],
        'PERCENT': sns.color_palette("Set2").as_hex()[6]
    }

    sns.barplot(
        x = 'counts',
        y = 'token',
        hue = 'ner_tag',
        palette = ner_palette,
        data = top_entities_fake[0:10],
        orient = 'h',
        dodge=False
    ) \
    .set(title='Most Common Entities in Fake News')

    plt.show()

    sns.barplot(
        x = 'counts',
        y = 'token',
        hue = 'ner_tag',
        palette = ner_palette,
        data = top_entities_fact[0:10],
        orient = 'h',
        dodge=False
    ) \
    .set(title='Most Common Entities in Factual News')

    # More names are used in fake news, while more locations are used in factual news

    plt.show()

    heading("Text Preprocessing")

    # a lot of the factual news has a location tag at the beginning of the article, let's use regex to remove this
    # Remove Belfast, London -, etc. from the beginning of the text


    # Remove everything before - (hyphen) and the hyphen itself
    data['text_clean'] = data.apply(lambda x: re.sub(r"^[^-]*-\s*", "", x['text']), axis=1)

    # lowercase 
    data['text_clean'] = data['text_clean'].str.lower()

    # remove punctuation
    data['text_clean'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['text_clean']), axis=1)

    # stop words
    en_stopwords = stopwords.words('english')
    heading2("Stop Words",en_stopwords) # check this against our most frequent n-grams

    data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (en_stopwords)]))

    # tokenize 
    data['text_clean'] = data.apply(lambda x: word_tokenize(x['text_clean']), axis=1)

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    data["text_clean"] = data["text_clean"].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

    heading2("data.head after preprocessing", data.head(10))

    # most common unigrams after preprocessing
    tokens_clean = sum(data['text_clean'], [])
    unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index()[:10]

    heading2("Unigrams",unigrams)



    unigrams['token'] = unigrams['index'].apply(lambda x: x[0]) # extract the token from the tuple so we can plot it

    sns.barplot(x = "count", 
                y = "token", 
                data=unigrams,
                orient = 'h',
                palette=[default_plot_colour],
                hue = "token", legend = False)\
    .set(title='Most Common Unigrams After Preprocessing')

    plt.show()


    bigrams = (pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()).reset_index()
    bigrams.columns = ['bigram', 'count']
    bigrams['token'] = bigrams['bigram'].apply(lambda x: ' '.join(x))  # join the tuple for display

    sns.barplot(
        x="count",
        y="token",
        data=bigrams[:10],
        orient='h',
        palette=[default_plot_colour],
        hue="token",
        legend=False
    ).set(title='Most Common Bigrams After Preprocessing')

    plt.show()


    heading("Sentiment Analysis")


    # use vader so we also get a neutral sentiment count
    vader_sentiment = SentimentIntensityAnalyzer()

    data['vader_sentiment_score'] = data['text'].apply(lambda review: vader_sentiment.polarity_scores(review)['compound'])

    heading2("data.head with sentiment scores", data.head(10))

    # create labels
    bins = [-1, -0.1, 0.1, 1]
    names = ['negative', 'neutral', 'positive']

    # pd.cut is used to segment and sort data values into bins
    # it is useful for converting continuous data into categorical data
    # here we are creating sentiment labels based on the vader sentiment score
    data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'], bins, labels=names)

    heading2("data.head with sentiment labels", data.head(10))

    data['vader_sentiment_label'].value_counts().plot.bar(color=default_plot_colour)
    plt.show()



    sns.countplot(
        x = 'fake_or_factual',
        hue = 'vader_sentiment_label',
        palette = sns.color_palette("hls"),
        data = data
    ) \
    .set(title='Sentiment by News Type')

    plt.show()

    heading("Topic Modelling")

    # fake news data vectorization
    fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].reset_index(drop=True)
    dictionary_fake = corpora.Dictionary(fake_news_text)
    doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

    # generate coherence scores to determine an optimum number of topics
    coherence_values = []
    model_list = []

    min_topics = 2
    # original max_topics = 11
    max_topics = 3

    for num_topics_i in range(min_topics, max_topics+1):
        model = gensim.models.LdaModel(doc_term_fake, num_topics=num_topics_i, id2word = dictionary_fake)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
        
    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

    
    # create lda model
    num_topics_fake = 2 
    #num_topics_fake = 6 

    lda_model_fake = gensim.models.LdaModel(corpus=doc_term_fake,
                                           id2word=dictionary_fake,
                                           num_topics=num_topics_fake)

    lda_model_fake.print_topics(num_topics=num_topics_fake, num_words=10)


    heading("Latent Semantic Analysis (LSA)")

    # our topics contain a lot of very similar words, let's try using latent semantic anaysis with tf-idf vectorization

    def tfidf_corpus(doc_term_matrix):
        # create a corpus using tfidf vecotization
        tfidf = TfidfModel(corpus=doc_term_matrix, normalize=True)
        corpus_tfidf = tfidf[doc_term_matrix]
        return corpus_tfidf

    def get_coherence_scores(corpus, dictionary, text, min_topics, max_topics):
        # generate coherence scores to determine an optimum number of topics
        coherence_values = []
        model_list = []
        for num_topics_i in range(min_topics, max_topics+1):
            model = LsiModel(corpus, num_topics=num_topics_i, id2word = dictionary, random_seed=0)
            model_list.append(model)
            coherence_model = CoherenceModel(model=model, texts=text, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherence_model.get_coherence())
        # plot results
        plt.plot(range(min_topics, max_topics+1), coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    heading2("Creating tfidf representation for fake news data and getting coherence scores", "Check")
    # create tfidf representation
    corpus_tfidf_fake = tfidf_corpus(doc_term_fake)

    heading2("corpus_tfidf_fake", corpus_tfidf_fake)
    # coherence scores for fake news data
    get_coherence_scores(corpus_tfidf_fake, dictionary_fake, fake_news_text, min_topics=2, max_topics=3)

    heading2("Creating LSA model for fake news data", "Check")

    # model for fake news data
    lsa_fake = LsiModel(corpus_tfidf_fake, id2word=dictionary_fake, num_topics=2)
    heading2("lsa_fake", lsa_fake)
    lsa_fake.print_topics()

    heading2("THE END - Fake News Classification", "Check")

    
    heading("Creating our own custom classification model")
    heading2("data.head before classification", data.head(10))
    

    X = [','.join(map(str, l)) for l in data['text_clean']]
    Y = data['fake_or_factual']

    # text vectorization - CountVectorizer
    countvec = CountVectorizer()
    countvec_fit = countvec.fit_transform(X)
    bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())
    heading2("bag_of_words.head", bag_of_words.head(10))


    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.3)

    lr = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)


    heading2("y_pred_lr", y_pred_lr)
    heading2("y_test", y_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_pred_lr, y_test))

    

    print(classification_report(y_test, y_pred_lr))

    svm = SGDClassifier().fit(X_train, y_train)

    y_pred_svm = svm.predict(X_test)

    heading2("y_pred_svm", y_pred_svm)
    heading2("y_test", y_test)
    print("SVM Accuracy:", accuracy_score(y_pred_svm, y_test))  # SVM is a type of linear classifier    

    print(classification_report(y_test, y_pred_svm))

if __name__ == "__main__":
    main()