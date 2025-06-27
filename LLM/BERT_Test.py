# BERT stands for Bidirectional Encoder Representations from Transformers, a transformer-based model for NLP tasks.
# It uses a transformer architecture to process text data and generate contextualized word embeddings.
# BERT is good for tasks like sentiment analysis, question answering, and named entity recognition.

# BERT was created by Google and is one of the most popular models in NLP.
# It is pre-trained on a large corpus of text and can be fine-tuned for specific tasks.
# BERT is a powerful model that can understand the context of words in a sentence, making
# it suitable for a wide range of NLP tasks.


# BERT is pre-trained on a large corpus of text and can be fine-tuned for specific tasks.
# It is a powerful model that can understand the context of words in a sentence, making it

# Popuplar on NLP tasks like sentiment analysis, question answering, and named entity recognition.

# BERT Base is a smaller version of BERT with 110 million parameters, while BERT Large has 345 million parameters.
# BERT Base is faster and requires less memory, while BERT Large is more accurate but
# requires more memory and computational resources.

# BERT has bidirectional attention, meaning it can consider the context of a word from both the left and right sides.
# This allows BERT to understand the meaning of a word in a sentence better than models that only consider the left context.

# GPT is auto-regressive, meaning it generates text one word at a time, while BERT is auto-encoding, meaning it processes the entire input at once.
# BERT is better suited for tasks that require understanding the context of a sentence, while GPT
# is better suited for tasks that require generating text.

# BERT looks at words to the left and right of the missing word to predict it, while GPT only looks at the words to the left.
# BERT is better at understanding the context of a sentence, while GPT is better at generating

# GPT models do well for conversational tasks, while BERT models do well for understanding the context of a sentence.

# BERT Architecture is based on the Transformer model, which uses self-attention mechanisms to process input sequences.

# BERT only uses the encoder part of the Transformer model, while GPT uses the decoder part.
# BERT is trained on a large corpus of text using a masked language modeling objective, where some words in the input are replaced with a mask token and the model is trained to predict the original words.
# GPT is trained on a large corpus of text using an auto-regressive language modeling objective, where the model is trained to predict the next word in a sequence given the previous words.
# BERT is a pre-trained model that can be fine-tuned for specific tasks, while GPT is a generative model that can generate text based on a given prompt.

# BERT consists of a stack of identical encoders, each containing a multi-head self-attention mechanism and a feed-forward neural network.
# BERT Base has 12 encoders, while BERT Large has 24 encoders.

# BERT uses 3 types of token embeddings: token embeddings, segment embeddings, and position embeddings.
# Token embeddings represent the words in the input sequence, segment embeddings represent the different segments in the input (e.g., question and answer), and position embeddings represent the position of each word in the input sequence.
# Segment embeddings are used to distinguish between different segments in the input, such as the question and answer in a question-answering task.
# Position embeddings are used to represent the position of each word in the input sequence, allowing the model to understand the order of words in a sentence.

# BERT has two objectives: masked language modeling and next sentence prediction.
# The masked language modeling objective is used to pre-train the model by randomly masking some words in the input and training the model to predict the original words.
# The next sentence prediction objective is used to train the model to understand the relationship between sentences,
# which is useful for tasks like question answering and natural language inference.

# BERT can predict the next sentence in a sequence, which is useful for tasks like question answering and natural language inference.

# Having no decoder makes BERT unsuitable for text generation tasks, as it cannot generate new text based on a given prompt.

import os
import sys
sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole
clearConsole()

heading("BERT Test")

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

model = BertForQuestionAnswering.from_pretrained(model_name)

tokenizer = BertTokenizer.from_pretrained(model_name)

# example question and text containing the answer
question = "When was the first dvd released?"
answer_document = "The first DVD (Digital Versatile Disc) was released on March 24, 1997. It was a movie titled 'Twister' and was released in Japan. DVDs quickly gained popularity as a replacement for VHS tapes and became a common format for storing and distributing digital video and data."

heading2("Question", question)
heading2("Answer Document", answer_document)
heading("Embedding the Question and Answer Document")

encoding = tokenizer.encode_plus(text=question, text_pair=answer_document)

heading2("Encoding",encoding)

inputs = encoding['input_ids']
heading2("Input IDs", inputs)
sentence_embedding = encoding['token_type_ids']
heading2("Token Type IDs", sentence_embedding)
tokens = tokenizer.convert_ids_to_tokens(inputs)
heading2("Tokens", tokens)

heading2("tokenizer.decode(101)", tokenizer.decode(101))
heading2("tokenizer.decode(102)", tokenizer.decode(102))


output = model(input_ids = torch.tensor([inputs]), token_type_ids = torch.tensor([sentence_embedding]))
heading("Model Output")
heading2("Output", output)


start_index = torch.argmax(output.start_logits)
end_index = torch.argmax(output.end_logits)

heading2("Start Index", start_index)
heading2("End Index", end_index)

answer = ' '.join(tokens[start_index:end_index+1])
heading2("Answer", answer)


import matplotlib as plt
import seaborn as sns

s_scores = output.start_logits.detach().numpy().flatten()
heading2("Start Scores", s_scores)
e_scores = output.end_logits.detach().numpy().flatten()
heading2("End Scores", e_scores)

token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

heading2("Token Labels", token_labels)


ax = sns.barplot(x=token_labels, y=s_scores)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.pyplot.show()

ax = sns.barplot(x=token_labels, y=e_scores)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
ax.grid(True)
plt.pyplot.show()

heading("BERT Question Answering Bot")

sunset_motors_context = "Sunset Motors is a renowned automobile dealership that has been a cornerstone of the automotive industry since its establishment in 1978. Located in the picturesque town of Crestwood, nestled in the heart of California's scenic Central Valley, Sunset Motors has built a reputation for excellence, reliability, and customer satisfaction over the past four decades. Founded by visionary entrepreneur Robert Anderson, Sunset Motors began as a humble, family-owned business with a small lot of used cars. However, under Anderson's leadership and commitment to quality, it quickly evolved into a thriving dealership offering a wide range of vehicles from various manufacturers. Today, the dealership spans over 10 acres, showcasing a vast inventory of new and pre-owned cars, trucks, SUVs, and luxury vehicles. One of Sunset Motors' standout features is its dedication to sustainability. In 2010, the dealership made a landmark decision to incorporate environmentally friendly practices, including solar panels to power the facility, energy-efficient lighting, and a comprehensive recycling program. This commitment to eco-consciousness has earned Sunset Motors recognition as an industry leader in sustainable automotive retail. Sunset Motors proudly offers a diverse range of vehicles, including popular brands like Ford, Toyota, Honda, Chevrolet, and BMW, catering to a wide spectrum of tastes and preferences. In addition to its outstanding vehicle selection, Sunset Motors offers flexible financing options, allowing customers to secure affordable loans and leases with competitive interest rates."
heading2("Sunset Motors Context", sunset_motors_context)

def faq_bot(question):
    context = sunset_motors_context
    input_ids = tokenizer.encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx+1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    output = model(torch.tensor([input_ids]), token_type_ids = torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = ' '.join(tokens[answer_start:answer_end+1])
    else:
        print("I don't know how to answer this question, can you ask another one?")
    corrected_answer = ''
    for word in answer.split():
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    return corrected_answer

heading("FAQ Bot")
heading2("Where is the dealership located?", faq_bot("Where is the dealership located?"))
heading2("What make of cars are available?", faq_bot("what make of cars are available?"))
heading2("How large is the dealership?", faq_bot("how large is the dealership?"))



# Roberta and DistilBERT are two popular transformer-based models for natural language processing tasks.
# Roberta is a variant of BERT that uses a different training objective and is trained on a larger dataset.
heading("Roberta and DistilBERT Test")

# Roberta stands for Robustly Optimized BERT Pretraining Approach, and it is designed to improve the performance of BERT by using a more robust training approach.
# DistilBERT is a smaller, faster, and lighter version of BERT that retains 97% of BERT's language understanding while being 60% faster and using 40% less memory.
# Roberta is better suited for tasks that require understanding the context of a sentence, while DistilBERT is better suited for tasks that require generating text.
# Robert uses dynamic masking, which means that the masking pattern is changed during training, while BERT uses static masking, which means that the masking pattern is fixed.
# Roberta is considered an impprovement over BERT, as it achieves better performance on various NLP tasks while being more efficient.


from transformers import RobertaTokenizer, RobertaModel
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

from transformers import  DistilBertTokenizer, DistilBertModel 
model_name = "distilbert-base-uncased"
tokenizer =  DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# REDO previous code with Roberta and DistilBERT
