import spacy
from spacy import displacy
from spacy import tokenizer
import re
import os


# Testing Git.

def clear_console():
       os.system('cls' if os.name == 'nt' else 'clear')

def heading(h):
      print("\n" + "="*len(h))
      print(h.upper())  
      print("="*len(h) + "\n")     

clear_console()

# !pip install spacy 

nlp = spacy.load('en_core_web_sm')

heading("Named Entity Recognition with spaCy")
heading("Original Google Text")

google_text = "Google was founded on September 4, 1998, by computer scientists Larry Page and Sergey Brin while they were PhD students at Stanford University in California. Together they own about 14% of its publicly listed shares and control 56% of its stockholder voting power through super-voting stock. The company went public via an initial public offering (IPO) in 2004. In 2015, Google was reorganized as a wholly owned subsidiary of Alphabet Inc. Google is Alphabet's largest subsidiary and is a holding company for Alphabet's internet properties and interests. Sundar Pichai was appointed CEO of Google on October 24, 2015, replacing Larry Page, who became the CEO of Alphabet. On December 3, 2019, Pichai also became the CEO of Alphabet."
print(google_text)

spacy_doc = nlp(google_text)

print()
print(f"Number of tokens in the text: {len(spacy_doc)}")
print(f"Number of sentences in the text: {len(list(spacy_doc.sents))}")
print(f"Number of entities in the text: {len(spacy_doc.ents)}")

def entityReport(spacy_doc):
    heading("Entities found in the text:")
    print(f"\n{'Entity':25} : {'Label'}\n")
    for word in spacy_doc.ents:
        print(f"{word.text:25} : {word.label_}")


entityReport(spacy_doc)

def setOutputAsWorkingDirectory():    
    current_directory = os.getcwd()
    print("Current working directory:", current_directory)
    output_directory = os.path.join(current_directory, "Outputs")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print("Output directory created:", output_directory)
    os.chdir(output_directory)

def getHtmlFromDoc(spacy_doc):
    html = displacy.render(spacy_doc, style="ent", jupyter=False)
    return html

# This will render the entities in a Jupyter notebook
#displacy.render(spacy_doc, style="ent", jupyter=True, options={'ents': ['ORG', 'GPE', 'DATE', 'PERSON']})

def save_doc_to_ouputs(html, filename):
    import os
    output_directory = os.path.join(os.getcwd(), "Outputs")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    file_path = os.path.join(output_directory, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDocument saved to {file_path}\n")

heading("Cleaning the text, remove punctuation and lowercasing")

google_text_clean_no_punctuation = re.sub(r'[^\w\s]', '', google_text) # remove punctuation and lowercase
print(google_text_clean_no_punctuation)
spacy_doc_clean_no_punctuation = nlp(google_text_clean_no_punctuation)
entityReport(spacy_doc_clean_no_punctuation)

google_text_clean_no_punctuation_lowercased = google_text.lower() # remove punctuation and lowercase
print(google_text_clean_no_punctuation_lowercased)
spacy_doc_clean_no_punctuation_lowercased = nlp(google_text_clean_no_punctuation_lowercased)
entityReport(spacy_doc_clean_no_punctuation_lowercased)

html_original = getHtmlFromDoc(spacy_doc)
html_clean_no_punctuation = getHtmlFromDoc(spacy_doc_clean_no_punctuation)
html_clean_no_punctuation_lowercased = getHtmlFromDoc(spacy_doc_clean_no_punctuation_lowercased)

html = "<h1>Named Entities Recognition</h1>\n"
html += "<br/><h2>Original Text</h2><br/>\n"
html += html_original
html += "<br/><h2>Cleaned Text - Removed punctuation</h2>\n"
html += html_clean_no_punctuation
html += "<br/><h2>Cleaned Text - Removed punctuation and lowercased</h2>\n"
html += html_clean_no_punctuation_lowercased

save_doc_to_ouputs(html, "NamedEntitiesRecognition_entities.html")