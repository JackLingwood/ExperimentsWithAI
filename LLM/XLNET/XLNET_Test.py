# XLNET was developed by Google AI and Carnegie Mellon University.
# It is a generalized autoregressive pretraining model that outperforms BERT on several NLP tasks.
# XLNET Large is a large version of XLNET with 24 layers, 1024 hidden units, and 16 attention heads.
# It is a llm that can be used for various NLP tasks such as text classification, question answering, and language generation.
# XLNET Base is a smaller version of XLNET with 12 layers, 768 hidden units, and 12 attention heads.


# XLNET is decoder only model, so it does not have a special token for the beginning of a sequence like BERT's [CLS] token.
# XLNET learns to predict the probability of a word given the context of the words that come before and after it.
# XLNET can capture bidirectional context by using a permutation-based training objective.
# XLNET is better than BERT on tasks that require understanding the relationship between words in a sentence.

# XLNET uses permutation-based training to capture bidirectional context.
# XLNet is a generalized autoregressive pretraining model that outperforms BERT on several NLP tasks.
# XLNet is a transformer-based model that uses a permutation-based training objective to capture bidirectional context.

# XLNet is performant on a wide range of NLP tasks, including text classification, question answering, and language generation.

import os
import sys
sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole
clearConsole()

import pandas as pd
import numpy as np
from cleantext import clean
import re
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datasets 
import evaluate
import random

heading("Training XLNet for Emotion Classification")

# Change the current working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))


data_train = pd.read_csv('emotion-labels-train.csv') 
data_test = pd.read_csv('emotion-labels-test.csv')
data_val = pd.read_csv('emotion-labels-val.csv')


print("Data loaded successfully from CSV files.")


# data should be saved in a folder called 'emotions' which is saved in the same place as your notebook

heading2("Data Train", data_train.head(10))

print("Preprocessing the data...")

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


data = pd.concat([data_train, data_test, data_val], ignore_index=True)
#data['text_clean'] = data['text'].apply(lambda x: clean(x, no_emoji=True))
data['text_clean'] = data['text'].apply(lambda x: clean(x))
data['text_clean'] = data['text'].apply(lambda x: remove_emoji(clean(x)))
data['text_clean'] = data['text_clean'].apply(lambda x: re.sub('@[^\s]+', '', x))

heading2("Data", data.head(20))


data['label'].value_counts().plot(kind="bar")

import matplotlib.pyplot as plt
plt.show()

g = data.groupby('label')
data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

data['label'].value_counts().plot(kind="bar")
plt.show()

data['label_int'] = LabelEncoder().fit_transform(data['label'])

NUM_LABELS = 4

train_split, test_split = train_test_split(data, train_size = 0.8)
train_split, val_split = train_test_split(train_split, train_size = 0.9)

heading2("len(train_split)", len(train_split))
heading2("len(test_split)", len(test_split))
heading2("len(val_split)", len(val_split))


# Create a DatasetDict for the train and test splits
train_df = pd.DataFrame({
    "label": train_split.label_int.values,
    "text": train_split.text_clean.values
})

test_df = pd.DataFrame({
    "label": test_split.label_int.values,
    "text": test_split.text_clean.values
})

# Convert DataFrames to Hugging Face Datasets
train_df = datasets.Dataset.from_dict(train_df)
# Convert DataFrame to Hugging Face Dataset
test_df = datasets.Dataset.from_dict(test_df)

# Create a DatasetDict
dataset_dict = datasets.DatasetDict({"train":train_df, "test":test_df})

heading2("DatasetDict", dataset_dict)

heading("Tokenizing the dataset with XLNetTokenizer for Emotion Classification")

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding = "max_length", max_length = 128, truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

heading2("Tokenized Datasets", tokenized_datasets)

heading2("tokenized_datasets['train']['text'][0]", tokenized_datasets['train']['text'][0])
heading2("tokenized_datasets['train']['input_ids'][0]", tokenized_datasets['train']['input_ids'][0])
exit()


tokenizer.decode(5)

print(tokenized_datasets['train']['token_type_ids'][0])

print(tokenized_datasets['train']['attention_mask'][0])

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', 
                                                       num_labels=NUM_LABELS, 
                                                       id2label={0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'})

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch", num_train_epochs=3)

trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics)

trainer.train()

trainer.evaluate()

model.save_pretrained("fine_tuned_model")

fine_tuned_model = XLNetForSequenceClassification.from_pretrained("fine_tuned_model")

clf = pipeline("text-classification", fine_tuned_model, tokenizer=tokenizer)

rand_int = random.randint(0, len(val_split))
print(val_split['text_clean'][rand_int])
answer = clf(val_split['text_clean'][rand_int], top_k=None)
print(answer)





