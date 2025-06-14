

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

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer



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
lemmatizer = WordNetLemmatizer()


def ApplyTwoStemmers(tokens):
    data = {
        'Name': ['Word', 'Porter Stemmer', 'Snowball Stemmer','WordNet Lemmatizer',"Noun","Verb","Adjective","Adverb","Adjective Satellite"],
        'Score': []
    }
    for t in tokens:
        basic = lemmatizer.lemmatize(t)
        verb = lemmatizer.lemmatize(t,pos="v")
        noun = lemmatizer.lemmatize(t,pos="n")
        adjective = lemmatizer.lemmatize(t,pos="a")
        adverb = lemmatizer.lemmatize(t,pos="r")
        adjectiveSatellite = lemmatizer.lemmatize(t,pos="s")
        print(f"{t:15} : {ps.stem(t):15} : {ss.stem(t):15} : {noun:15} : {verb:15} : {adjective:15} : {adverb:15}  : {adjectiveSatellite:15}")
        data['Score'].append([t, ps.stem(t), ss.stem(t), basic, noun, verb, adjective, adverb, adjectiveSatellite])
    return data


def ShowSubPlots(data):
    df = pd.DataFrame(data['Score'], columns=data['Name'])
    fig, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.show()

def GenerateReadmeMDTableMarkdown(data):
    lines = []
    lines.append("| Word | Porter Stemmer | Snowball Stemmer | WordNet Lemmatizer | Noun             | Verb             | Adjective           | Adverb            | Adjective Satellite |")
    lines.append("|------|----------------|-------------------|-------------------| -----------------|------------------|---------------------|-------------------|---------------------|") 
    for row in data['Score']:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |")
    return "\n".join(lines)




def DemoGraphObjectsTable(data):
    # This function demonstrates how to create a table using Plotly's graph_objects
    import plotly.graph_objects as go

    df = pd.DataFrame(data['Score'], columns=data['Name'])

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
    'likes','better','worse',
    "running", "ate", "singing", "wrote", "driving", "flying", "went", "swimming",
    "spoke", "buying", "geese", "mice", "children", "feet", "teeth", "men", "women",
    "oxen", "leaves", "data", "better", "worse", "faster", "happier", "bigger"    
]

print("Stemming words using Porter Stemmer, Snowball Stemmer and WordNet Lemmatizer:")
print ("="*70)

# print Heading
print(f"{'Word':15} : {'Stemmed Word':15} : {'Snowball':15} : {'WordNetLemmatizer':15}")

data = ApplyTwoStemmers(words)
print("\nStemming complete.\n")
print(data)

DemoGraphObjectsTable(data)
ShowSubPlots(data) # Renders badly because there is too much data for the table to fit nicely
print(GenerateReadmeMDTableMarkdown(data))

#help(ps.stem) # Check out the help for the stem function