import os
import sys
sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole
clearConsole()

heading("Hugging Face Transformers")

from transformers import pipeline

sentiment_classifier = pipeline("sentiment-analysis")

sentiment_classifier("I'm so excited to be learning about large language models")

ner = pipeline("ner", model = "dslim/bert-base-NER")

basicNER = ner("Her name is Anna and she works in New York City for Morgan Stanley.")

print(basicNER)

# https://huggingface.co/models

zeroshot_classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")

sequence_to_classify = "one day I will see the world"
candidate_labels = ['travel', 'cooking', 'dancing']

zero = zeroshot_classifier(sequence_to_classify, candidate_labels)

print(zero)

print("Pretrained tokenizers")

from transformers import AutoTokenizer

model = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model)

sentence = "I'm so excited to be learning about large language models"
heading2("Sentence", sentence)

input_ids = tokenizer(sentence)

heading2("Input IDs", input_ids)

print("token type ids are used for tasks like question answering where you have a pair of sentences")
print("In this case, the input is a single sentence, so the token type ids are all 0")
print("tokens with type id 0 are from the first sentence, and tokens with type id 1 are from the second sentence")
print("The attention mask is used to specify which tokens should be attended to by the model")

tokens = tokenizer.tokenize(sentence)
heading2("Tokens", tokens)

# Convert tokens to IDs
# The tokenizer converts the tokens into their corresponding IDs


token_ids = tokenizer.convert_tokens_to_ids(tokens)
heading2("Token IDs", token_ids)

# Decode the token IDs back to a string
# The tokenizer can also decode the token IDs back to a string representation
# this is useful for understanding how the model interprets the input
# and for debugging purposes

decoded_ids = tokenizer.decode(token_ids)
heading2("Decoded IDs", decoded_ids)



print("Decoded IDs for special tokens")
print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)
print("PAD token:", tokenizer.pad_token)
print("Special token IDs")
print("CLS token ID:", tokenizer.cls_token_id)
print("SEP token ID:", tokenizer.sep_token_id)
print("PAD token ID:", tokenizer.pad_token_id)
print("Decoded IDs for special tokens")
print("CLS token ID:", tokenizer.decode(tokenizer.cls_token_id))
print("SEP token ID:", tokenizer.decode(tokenizer.sep_token_id))
print("PAD token ID:", tokenizer.decode(tokenizer.pad_token_id))
print("Decoded IDs for special tokens using IDs")
print(tokenizer.decode(tokenizer.cls_token_id))
print(tokenizer.decode(tokenizer.sep_token_id))
print(tokenizer.decode(tokenizer.pad_token_id))
print("Decoded IDs for special tokens using IDs")
print(tokenizer.decode(101))  # CLS token ID
print(tokenizer.decode(102))  # SEP token ID
print(tokenizer.decode(0))    # PAD token ID
print("Decoded IDs for special tokens using IDs")
print(tokenizer.decode(tokenizer.cls_token_id))  # CLS token ID
print(tokenizer.decode(tokenizer.sep_token_id))  # SEP token ID
print(tokenizer.decode(tokenizer.pad_token_id))  # PAD token ID
print("Decoded IDs for special tokens using IDs")

# SEP stands for separator, CLS stands for classification, PAD stands for padding

heading2("tokenizer.decode(101)", tokenizer.decode(101))  # CLS token ID
heading2("tokenizer.decode(102)", tokenizer.decode(102))  # SEP token ID
heading2("tokenizer.decode(0)", tokenizer.decode(0))    # PAD token ID

heading("Using XLNet for Tokenization - Another LLM Example")

heading2("Sentence", sentence)

model2 = "xlnet-base-cased"
heading2("Model for XLNet", model2)

tokenizer2 = AutoTokenizer.from_pretrained(model2)
input_ids = tokenizer2(sentence)
heading2("Input IDs for XLNet", input_ids)

tokens = tokenizer2.tokenize(sentence)
heading2("Tokens for XLNet", tokens)

token_ids = tokenizer2.convert_tokens_to_ids(tokens)
heading2("Token IDs for XLNet", token_ids)

# 4 = SEP token ID in XLNet
# 3 = CLS token ID in XLNet
heading2("tokenizer2.decode(4)", tokenizer2.decode(4))

heading2("tokenizer2.decode(3)", tokenizer2.decode(3))

heading("Speacial Tokens - Using DistilBERT for Sentiment Analysis with Torch")

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

heading2("Sentence", sentence)
heading2("Input IDs", input_ids)
print("Special tokens are used to indicate the start and end of a sequence, and to separate different sequences in a batch. ")
print("In this case, the input is a single sentence, so the special tokens are not used.")

# Special tokens are markers that help the model understand the structure of the input data.
# For example, in a sequence classification task, the special tokens are used to indicate the start and end of the sequence.
# CLS (classification) token is used to indicate the start of the sequence, and SEP (separator) token is used to separate different sequences in a batch.
# CLS is usually placed at the beginning of the sequence, and SEP is placed at the end of the sequence.
# The mask token is used to indicate which tokens should be attended to by the model.
# Mask tokens are used in tasks like masked language modeling, where some tokens in the input are replaced with a special mask token to predict the original token.
# In this case, the input is a single sentence, so the mask token is not used.
# Custom special tokens can be added to the tokenizer if needed, but in this case, the default special tokens are sufficient for the task.
# Special tokens for padding, classification, and separation are already defined in the tokenizer.
# Padding tokens are used to ensure that all sequences in a batch have the same length, which is necessary for efficient processing by the model.

# TensorFlow and PyTorch are two popular deep learning frameworks that can be used with Hugging Face Transformers.


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

input_ids_pt = tokenizer(sentence, return_tensors = "pt")

heading2("Input IDs for DistilBERT", input_ids_pt)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# The model is a pre-trained DistilBERT model fine-tuned for sentiment analysis on the SST-2 dataset.
# It is used to classify the sentiment of the input text as positive or negative.
# The model takes the input IDs as input and returns the logits, which are the raw predictions of the model.
# no_grad is used to disable gradient calculation, which is not needed for inference.
# This is useful for saving memory and speeding up the inference process.
with torch.no_grad():
    logits = model(**input_ids_pt).logits

heading2("Logits", logits)

predicted_class_id = logits.argmax().item()

heading2("Predicted Class ID", predicted_class_id)
heading2("Predicted Class Label", model.config.id2label[predicted_class_id])
model.config.id2label[predicted_class_id]

model_directory = "my_saved_models"

heading2("Saving the model and tokenizer to", model_directory)
tokenizer.save_pretrained(model_directory)

# Save the model to the specified directory

heading2("Saving the model to", model_directory)
model.save_pretrained(model_directory)

heading2("Loading the model and tokenizer from", model_directory)
my_tokenizer = AutoTokenizer.from_pretrained(model_directory)

heading2("Loading the model from", model_directory)
my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)