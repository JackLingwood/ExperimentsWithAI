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

from transformers import AutoTokenizer

model = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model)

sentence = "I'm so excited to be learning about large language models"

input_ids = tokenizer(sentence)
print(input_ids)

tokens = tokenizer.tokenize(sentence)

print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(token_ids)

decoded_ids = tokenizer.decode(token_ids)
print(decoded_ids)

tokenizer.decode(101)

tokenizer.decode(102)

model2 = "xlnet-base-cased"

tokenizer2 = AutoTokenizer.from_pretrained(model2)

input_ids = tokenizer2(sentence)

print(input_ids)

tokens = tokenizer2.tokenize(sentence)
print(tokens)

token_ids = tokenizer2.convert_tokens_to_ids(tokens)
print(token_ids)

tokenizer2.decode(4)

tokenizer2.decode(3)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

print(sentence)
print(input_ids)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

input_ids_pt = tokenizer(sentence, return_tensors ="pt")
print(input_ids_pt)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

with torch.no_grad():
    logits = model(**input_ids_pt).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

model_directory = "my_saved_models"

tokenizer.save_pretrained(model_directory)

model.save_pretrained(model_directory)

my_tokenizer = AutoTokenizer.from_pretrained(model_directory)

my_model = AutoModelForSequenceClassification.from_pretrained(model_directory)