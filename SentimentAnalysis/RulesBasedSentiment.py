
import sys
import os
sys.path.append(os.path.abspath("Shared"))  # Add the parent directory to the system path
print(sys.path)
from utils import heading, clearConsole
clearConsole()
heading("Rules Based Sentiment Analysis Practical")

# pip install textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_sentiment = SentimentIntensityAnalyzer()


sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"
sentence_5 = "i had the best time of my life the movie it was the best experience of my life"
sentence_6 = "i wish I could have the time back that I spent watching this movie"


sentences = [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5, sentence_6]

def evaluate_sentiment(sentence):
    sentiment = TextBlob(sentence)
    vader_sent = vader_sentiment.polarity_scores(sentence_2)
    print(f"Sentence: {sentence}")
    print(f"\tTextBlob Sentiment: {sentiment.sentiment.polarity}, \n\tVADER Sentiment:    {vader_sent["compound"]} - {vader_sent}")

for sentence in sentences:
    evaluate_sentiment(sentence)


heading("Sentiment Analysis with TextBlob")


print(sentence_1)
sentiment_score = TextBlob(sentence_1)
print(sentiment_score.sentiment.polarity)

print("\n[Score is -1 to 1, where -1 is negative, 0 is neutral and 1 is positive]\n")


print(sentence_2)
sentiment_score_2 = TextBlob(sentence_2)
print(sentiment_score_2.sentiment.polarity)

print(sentence_3)
sentiment_score_3 = TextBlob(sentence_3)
print(sentiment_score_3.sentiment.polarity)

print(sentence_4)
sentiment_score_4 = TextBlob(sentence_4)
print(sentiment_score_4.sentiment.polarity)

heading("Sentiment Analysis with VADER")



print(sentence_1)
print(vader_sentiment.polarity_scores(sentence_1))

print(sentence_2)
print(vader_sentiment.polarity_scores(sentence_2)) 

print(sentence_3)
print(vader_sentiment.polarity_scores(sentence_3)) 

print(sentence_4)
print(vader_sentiment.polarity_scores(sentence_4)) 

# pip install transformers
import transformers
from transformers import pipeline


# What is list of models available?
def list_transformer_models():
    heading("Available Transformer Models for Sentiment Analysis")
    from huggingface_hub import list_models
    #models = list_models(filter="task:sentiment-analysis")
    models = list_models()
    #models = list_models(filter="pipeline_tag:sentiment-analysis,library:transformers")
    #models = list_models(filter="pipeline_tag:sentiment-analysis")

    x = 1

    for model in models:  # show first 10
        #print(model.modelId)
        print(f"Model ID: {model.modelId}, Tags: {model.tags}, Pipeline Tags: {model.pipeline_tag}")
        x += 1
        if x > 10:
            break


# There are many models available, so we will just list a few.
list_transformer_models()

heading("Sentiment Analysis with Transformers")

def evaluate_transformer_sentiment(sentences, model):
    heading(f"Sentiment Analysis with Transformers with model {model}")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model)
    for sentence in sentences:
        print(f"Sentence: {sentence}")
        print(f"Transformer Sentiment: {sentiment_pipeline(sentence)}")

evaluate_transformer_sentiment(sentences, r"finiteautomata/bertweet-base-sentiment-analysis")
evaluate_transformer_sentiment(sentences, r"nlptown/bert-base-multilingual-uncased-sentiment")
evaluate_transformer_sentiment(sentences, r"finiteautomata/bertweet-base-sentiment-analysis")
evaluate_transformer_sentiment(sentences, r"cardiffnlp/twitter-roberta-base-sentiment")
evaluate_transformer_sentiment(sentences, r"distilbert-base-uncased-finetuned-sst-2-english")
evaluate_transformer_sentiment(sentences, r"nlptown/bert-base-multilingual-uncased-sentiment")
evaluate_transformer_sentiment(sentences, r"cardiffnlp/twitter-roberta-base-sentiment-latest")
evaluate_transformer_sentiment(sentences, r"siebert/sentiment-roberta-large-english")
evaluate_transformer_sentiment(sentences, r"boltuix/bert-emotion")








