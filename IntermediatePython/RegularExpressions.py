# Regular Expressions (Regex)
# Regular expressions are a powerful tool for searching and manipulating strings.
# They allow you to define patterns that can match specific sequences of characters.
# This code demonstrates the use of regular expressions in Python.

import re
#help(re)
#   https://docs.python.org/3.12/library/re.html

print('Write r for raw strings, r"string"')

# r"STRING" --> RAW STRING
my_folder = r"C:\desktop\notes";
print(my_folder)


# Match string
result_match = re.match("pattern", r"string to contain the pattern")
print(result_match)

# Match string at the beginning
result_match_2 = re.match("pattern", r"the pattern is at the beginning of this string")
print(result_match_2)

result_search_2 = re.search("pattern", r"the phrase to find isn't in this string")
print(result_search_2)

# Replace string

print("Use re.sub() to replace a string")
help(re.sub)
string = r"sara was help me to find the items i needed quickly"
new_string = re.sub("sara","sarah",string)
print(new_string)

customer_reviews = ["same was a great help to me in the store",
                   "the cashier was very rude to me, I think her name was eleanor",
                   "amazing work from sadeen!",
                   "sarah was able to help me find the items I needed quickly",
                   "lucy is such a great addition to the team",
                   "great service from sara she found me what i wanted"]

sarahs_reviews = []

pattern_to_find = r"sarah?" # ?=h is optional

for string in customer_reviews:
    if(re.search(pattern_to_find,string)):
        sarahs_reviews.append(string)

print(sarahs_reviews)

a_reviews = []
pattern_to_find = r"^a" # Find all strings that start with a, ^ = START OF STRING

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        a_reviews.append(string)

print(a_reviews)

y_reviews = []
pattern_to_find = r"y$" # Find all strings that end with a y, $ = END OF STRING
for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        y_reviews.append(string)

print(y_reviews)

needwant_reviews = []
pattern_to_find = r"(need|want)ed" # Find all string with need OR want, | (pipe) = OR

for string in customer_reviews:
    if (re.search(pattern_to_find, string)):
        needwant_reviews.append(string)

print(needwant_reviews)

no_punct_reviews = []
pattern_to_find = r"[^\w\s]"    # [^xxxx] = NOT xxxx, \w = WORD, \s = WHITESPACE
#important to remove punctuation for NLP

for string in customer_reviews:
    no_punct_string = re.sub(pattern_to_find, "", string)
    no_punct_reviews.append(no_punct_string)    

print(no_punct_reviews)