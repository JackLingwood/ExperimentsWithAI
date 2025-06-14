

# Stemming = Change word to base word
# Stemming is part of Pre-processing text
# Words are reduced to their base form
# Stemming removes suffix of word,
# But sometimes produces nonsense words
# Stemming reduces size and complexity of data
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
import pandas as pd
import matplotlib.pyplot as plt
# nltk.download('punkt') # Uncomment if you need to download the punkt tokenizer
# nltk.download('averaged_perceptron_tagger') # Uncomment if you need to download the POS tagger
# nltk.download('wordnet') # Uncomment if you need to download the WordNet lemmatizer
# nltk.download('omw-1.4') # Uncomment if you need to download the Open Multilingual WordNet
# nltk.download('stopwords') # Uncomment if you need to download the stopwords corpus
# nltk.download('maxent_ne_chunker') # Uncomment if you need to download the named entity chunker
# nltk.download('words') # Uncomment if you need to download the words corpus
# nltk.download('punkt') # Uncomment if you need to download the punkt tokenizer
# nltk.download('averaged_perceptron_tagger') # Uncomment if you need to download the POS tagger
# nltk.download('wordnet') # Uncomment if you need to download the WordNet lemmatizer
# nltk.download('omw-1.4') # Uncomment if you need to download the Open Multilingual WordNet




ps = PorterStemmer() # Needs to be initialized
ss = SnowballStemmer('english') # Needs to be initialized



def ApplyTwoStemmers(tokens):
    data = {
        'Name': ['Word', 'Porter Stemmer', 'Snowball Stemmer'],
        'Score': []
    }
    for t in tokens:
        print(f"{t:15} : {ps.stem(t):15} : {ss.stem(t):15}")
        data['Score'].append([t, ps.stem(t), ss.stem(t)])
    return data


def ShowSubPlots(data):
    df = pd.DataFrame(data['Score'], columns=data['Name'])
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.show()

def GenerateReadmeMDTableMarkdown(data):
    lines = []
    lines.append("| Word | Porter Stemmer | Snowball Stemmer |")
    lines.append("|------|----------------|-------------------|")
    for row in data['Score']:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    return "\n".join(lines)




def DemoGraphObjectsTable(df2):
    # This function demonstrates how to create a table using Plotly's graph_objects
    import plotly.graph_objects as go

    # Create a DataFrame
    # df = pd.DataFrame({
    #     "Name": ["Alice", "Bob", "Charlie"],
    #     "Score": [85, 90, 95]
    # })

    # df = pd.DataFrame({
    #     "Name": [f"Name {i}" for i in range(100)],
    #     "Score": [i for i in range(100)]
    # })
    df = pd.DataFrame(df2['Score'], columns=df2['Name'])

    print(df)

    # Create a table figure
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])

    fig.update_layout(
    height=1000,  # Control vertical size
    )

    fig.show()



# import plotly.graph_objects as go

# # Create a large DataFrame
# df = pd.DataFrame({
#     "Name": [f"Name {i}" for i in range(100)],
#     "Score": [i for i in range(100)]
# })

# # Create a scrollable table
# fig = go.Figure(data=[go.Table(
#     header=dict(values=list(df.columns)),
#     cells=dict(values=[df[col] for col in df.columns])
# )])

# fig.update_layout(
#     height=600,  # Control vertical size
# )

# fig.show()






# df = pd.DataFrame(data)

# fig, ax = plt.subplots()
# ax.axis('off')
# table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
# plt.show()



# print string with right padding
#def print_stemmed(tokens):
 #   for t in tokens:
  #      print(f"{t:15} : {ps.stem(t)}")

words = [
    "connect", "connected", "connecting", "connection", "connections",
    'connectivity','connects',
    "operate", "operated", "operates", "operating", "operation", "operations",
    "relate", "related", "relates", "relating", "relation", "relations",
    "create", "created", "creates", "creating", "creation", "creations",
    "analyze", "analyzed", "analyzes", "analyzing", "analysis", "analyses",
    'learned','learning','learn','learns','learner','learners',
    "run", "running", "runner", "ran",
    "fly", "flying", "flew", "flies",
    "study", "studies", "studying", "studied",
    "happy", "happier", "happiest", "happiness",
    "use", "used", "uses", "using", "useful", "useless",
    'likes','better','worse'
]

print("Stemming words using Porter Stemmer:")
print ("="*50)

# print Heading
print(f"{'Word':15} : {'Stemmed Word':15} : {'Snowball':15}")


#from nltk.tokenize import word_tokenize
data = ApplyTwoStemmers(words)
print("\nStemming complete.\n")
print(data)

DemoGraphObjectsTable(data)
print(GenerateReadmeMDTableMarkdown(data))
#ShowSubPlots(data)



#help(ps.stem) # Check out the help for the stem function


# worse: wors --> WARNING: wors is NOT a word

# Lemmatization - stems word to more meaningful base form than stemming
# Works more intelligently than stemming
# # Keeps meaning but leaves you with MUCH larger dataset
# # Reduces words to their base form
# import nltk
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer

# lemmatizer = WordNetLemmatizer()

# for t in connect_tokens:
#     print(t,": ",lemmatizer.lemmatize(t))

# for t in learn_tokens:
#     print(t,": ",lemmatizer.lemmatize(t))

# for t in likes_tokens:
#     print(t,": ",lemmatizer.lemmatize(t))