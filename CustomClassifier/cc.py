# Custom Text Classifier
# Powerful
# Classifying text data is a common task in natural language processing (NLP) and machine learning.

# We are creating a custom text classifier that can categorize text documents into predefined classes or labels based on their content.
# Custom classifiers are built to handle specific text classification tasks, such as sentiment analysis, spam detection, or topic classification.
# Custom classifiers are trained on labeled text data, where each document is associated with a specific class or label.
# Custom classifiers can be used to predict the class or label of




# It involves assigning predefined categories or labels to text documents based on their content.
# Custom classifiers can be built using various machine learning algorithms, such as Naive Bayes, Support Vector Machines (SVM), or deep learning models like recurrent neural networks (RNNs) or transformers.
# Custom classifiers can be trained on labeled text data to learn patterns and relationships between the text and the corresponding labels.
# The trained classifier can then be used to predict the labels of new, unseen text documents.

# This is a supervised Machine Learning task where the model learns from labeled data to make predictions on new, unseen data.

# We will use three different approaches to build a custom text classifier:
# 1. Logistic Regression
# 2. Naive Bayes
# 3. Linear Support Vector Machine (SVM)

# 365 Data Science Platform trains how optimize a custom text classifier using the Bag of Words model and various machine learning algorithms.
# https://365datascience.com/courses/machine-learning-fundamentals/custom-text-classifier/


import os
import sys

sys.path.append(os.path.abspath("Shared"))
from utils import heading, clearConsole
clearConsole()

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.DataFrame([("i love spending time with my friends and family", "positive"),
                     ("that was the best meal i've ever had in my life", "positive"),
                     ("i feel so grateful for everything i have in my life", "positive"),
                     ("i received a promotion at work and i couldn't be happier", "positive"),
                     ("watching a beautiful sunset always fills me with joy", "positive"),
                     ("my partner surprised me with a thoughtful gift and it made my day", "positive"),
                     ("i am so proud of my daughter for graduating with honors", "positive"),
                     ("listening to my favorite music always puts me in a good mood", "positive"),
                     ("i love the feeling of accomplishment after completing a challenging task", "positive"),
                     ("i am excited to go on vacation next week", "positive"),
                     ("i feel so overwhelmed with work and responsibilities", "negative"),
                     ("the traffic during my commute is always so frustrating", "negative"),
                     ("i received a parking ticket and it ruined my day", "negative"),
                     ("i got into an argument with my partner and we're not speaking", "negative"),
                     ("i have a headache and i feel terrible", "negative"),
                     ("i received a rejection letter for the job i really wanted", "negative"),
                     ("my car broke down and it's going to be expensive to fix", "negative"),
                     ("i'm feeling sad because i miss my friends who live far away", "negative"),
                     ("i'm frustrated because i can't seem to make progress on my project", "negative"),
                     ("i'm disappointed because my team lost the game", "negative")
                    ],
                    columns=['text', 'sentiment'])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

X = data['text']
y = data['sentiment']

heading("Text Data and Sentiments")
print(X)
print(y)
print("Number of samples:", len(X))




# text vectorization to bow - CountVectorizer
countvec = CountVectorizer()
countvec_fit = countvec.fit_transform(X)
bag_of_words = pd.DataFrame(countvec_fit.toarray(), columns = countvec.get_feature_names_out())

heading("Bag of Words Model - CountVectorizer")
print(bag_of_words)

# split into train and test data
# 30% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, y, test_size=0.3, random_state = 7)

heading("Train and Test Data")
print("Train Data Shape:", X_train.shape)
print("Test Data Shape:", X_test.shape)
print("Train Sentiments Shape:", y_train.shape)
print("Test Sentiments Shape:", y_test.shape)
print("Train Data Sample:\n", X_train.head())
print("Test Data Sample:\n", X_test.head())





heading("Train and Test Data - Logicistic Regression")

lr = LogisticRegression(random_state=1).fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("Logistic Regression Model Accuracy:", accuracy_score(y_pred_lr, y_test))
# Accuraccy of 0.166 is very low, but this is expected as we have a very small dataset and the model is not trained well.

print("Predicted Sentiments:", y_pred_lr)
print("Actual Sentiments:", y_test)
print("Accuracy Score:", accuracy_score(y_pred_lr, y_test))

print(classification_report(y_test, y_pred_lr, zero_division=0))

# Naive Bayes Classifier
heading("Naive Bayes Classifier")


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB().fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)

print("Naive Bayes Model Accuracy:", accuracy_score(y_pred_nb, y_test))

accuracy_score(y_pred_nb, y_test)

print(classification_report(y_test, y_pred_nb, zero_division=0))



heading("Linear Support Vector Machine Classifier")
from sklearn.linear_model import LogisticRegression, SGDClassifier
svm = SGDClassifier().fit(X_train, y_train)
# possible hyper params, loss function, regularization
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine Model Accuracy:", accuracy_score(y_pred_svm, y_test))
accuracy_score(y_pred_svm, y_test)
print(classification_report(y_test, y_pred_svm, zero_division=0))