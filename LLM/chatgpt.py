# pip install openai


import os
import sys

sys.path.append(os.path.abspath("Shared"))
from utils import heading, heading2, clearConsole

clearConsole()

# Change current working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("Current working directory:", os.getcwd())


import config
heading("ChatGPT API")
api_key = config.api_key
#heading2("api_key", api_key)

#import openai
# Function to get a response from ChatGPT
from openai import OpenAI

def getChatGPTResponse(prompt, temperature=0.2, max_tokens=10):
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(    
    model="gpt-3.5-turbo",
    max_tokens=max_tokens,
    temperature=temperature,
    messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content


def getChatGPTResponse(specialMessages, temperature=0.2, max_tokens=10):
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=specialMessages
    )
    return chat_completion.choices[0].message.content


# Max tokens and temperature can be adjusted as needed
# Example usage
# max_tokens = 10 will limit the response to 10 tokens
# temperature = 0.2 will make the response more deterministic
# You can change these values based on your requirements
# max_tokens = 10
# temperature = 0
# How long is a token?
# A token is approximately 4 characters of English text, so 100 tokens is about 75 words.
# Example prompt



#response = getChatGPTResponse("Charlize Theron is an actress and producer. What are some of her most famous movies?")
#print(response)

#resonse = getChatGPTResponse("Once upon a time, in a land far, far away", 0.2, 100)
#print(resonse)


# Inside messages the role can be "system", "user", or "assistant"
# The "system" role is used to set the behavior of the assistant
# The "user" role is used to provide input to the assistant
# The "assistant" role is used to provide the assistant's response
# Example of using the system role to set the behavior of the assistant


def text_summarizer(prompt):
    client = OpenAI(api_key=api_key)
    chat_completion = client.chat.completions.create(    
    model="gpt-3.5-turbo",
    messages=[
        {
          "role": "system",
          "content": "You will be provided with a block of text, and your task is to extract a list of keywords from it."
        },
        {
          "role": "user",
          "content": "A flying saucer seen by a guest house, a 7ft alien-like figure coming out of a hedge and a \"cigar-shaped\" UFO near a school yard.\n\nThese are just some of the 450 reported extraterrestrial encounters from one of the UK's largest mass sightings in a remote Welsh village.\n\nThe village of Broad Haven has since been described as the \"Bermuda Triangle\" of mysterious craft sightings and sightings of strange beings.\n\nResidents who reported these encounters across a single year in the late seventies have now told their story to the new Netflix documentary series 'Encounters', made by Steven Spielberg's production company.\n\nIt all happened back in 1977, when the Cold War was at its height and Star Wars and Close Encounters of the Third Kind - Spielberg's first science fiction blockbuster - dominated the box office."
        },
        {
          "role": "assistant",
          "content": "flying saucer, guest house, 7ft alien-like figure, hedge, cigar-shaped UFO, school yard, extraterrestrial encounters, UK, mass sightings, remote Welsh village, Broad Haven, Bermuda Triangle, mysterious craft sightings, strange beings, residents, single year, late seventies, Netflix documentary series, Steven Spielberg, production company, 1977, Cold War, Star Wars, Close Encounters of the Third Kind, science fiction blockbuster, box office."
        },
        {
          "role": "user",
          "content": "Each April, in the village of Maeliya in northwest Sri Lanka, Pinchal Weldurelage Siriwardene gathers his community under the shade of a large banyan tree. The tree overlooks a human-made body of water called a wewa – meaning reservoir or \"tank\" in Sinhala. The wewa stretches out besides the village's rice paddies for 175-acres (708,200 sq m) and is filled with the rainwater of preceding months.    \n\nSiriwardene, the 76-year-old secretary of the village's agrarian committee, has a tightly-guarded ritual to perform. By boiling coconut milk on an open hearth beside the tank, he will seek blessings for a prosperous harvest from the deities residing in the tree. \"It's only after that we open the sluice gate to water the rice fields,\" he told me when I visited on a scorching mid-April afternoon.\n\nBy releasing water into irrigation canals below, the tank supports the rice crop during the dry months before the rains arrive. For nearly two millennia, lake-like water bodies such as this have helped generations of farmers cultivate their fields. An old Sinhala phrase, \"wewai dagabai gamai pansalai\", even reflects the technology's centrality to village life; meaning \"tank, pagoda, village and temple\"."
        },
        {
          "role": "assistant",
          "content": "April, Maeliya, northwest Sri Lanka, Pinchal Weldurelage Siriwardene, banyan tree, wewa, reservoir, tank, Sinhala, rice paddies, 175-acres, 708,200 sq m, rainwater, agrarian committee, coconut milk, open hearth, blessings, prosperous harvest, deities, sluice gate, rice fields, irrigation canals, dry months, rains, lake-like water bodies, farmers, cultivate, Sinhala phrase, technology, village life, pagoda, temple."
        }, 
        {
          "role": "user",
          "content": prompt
        }
      ],
          temperature=0.5,
        max_tokens=256
      ) 
    return chat_completion.choices[0].message.content.strip()


prompt = "Master Reef Guide Kirsty Whitman didn't need to tell me twice. Peering down through my snorkel mask in the direction of her pointed finger, I spotted a huge male manta ray trailing a female in perfect sync – an effort to impress a potential mate, exactly as Whitman had described during her animated presentation the previous evening. Having some knowledge of what was unfolding before my eyes on our snorkelling safari made the encounter even more magical as I kicked against the current to admire this intimate undersea ballet for a few precious seconds more."
print(prompt)

text_summarizer(prompt)


def poetic_chatbot(prompt):
        # This function creates a poetic chatbot that responds to user queries in a poetic manner.
        # It uses the OpenAI API to generate responses based on a set of predefined messages.
        # The messages include a system message to set the behavior of the assistant, user messages to guide the assistant's responses, and assistant messages that provide poetic answers.
        # The temperature and max_tokens parameters can be adjusted to control the creativity and length of the responses.
        messages = [
            {
                "role": "system", # Overall system message to set the behavior of the assistant # Most important message
                "content": "You are a poetic chatbot."
            },
            {
                "role": "user", # Give examples of user messages to guide the assistant's responses
                "content": "When was Google founded?"
            },
            {
                "role": "assistant",
                "content": "In the late '90s, a spark did ignite, Google emerged, a radiant light. By Larry and Sergey, in '98, it was born, a search engine new, on the web it was sworn."
            },
            {
                "role": "user",
                "content": "Which country has the youngest president?"
            },
            {
                "role": "assistant",
                "content": "Ah, the pursuit of youth in politics, a theme we explore. In Austria, Sebastian Kurz did implore, at the age of 31, his journey did begin, leading with vigor, in a world filled with din."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        return getChatGPTResponse(messages, temperature=0.5, max_tokens=256)


heading("Poetic Chatbot")
prompt = "When was cheese first made?"
heading2(prompt, poetic_chatbot(prompt))
prompt = "What is the next course to be uploaded to 365DataScience?"
heading2(prompt, poetic_chatbot(prompt))
# Above answer is not up-to-date.

# We can use LangChain to create a conversational retrieval chain that can answer questions based on a specific set of documents.
# LangChain is a framework for building applications with LLMs (Large Language Models) like ChatGPT.
# It allows us to create chains of operations that can include loading documents, splitting them into smaller
# chunks, embedding them into a vector store, and then using a conversational retrieval chain to answer questions based on those documents.

# We can use LangChain to generate a support chatbot.

# We use LangChain to to add additional up-to-date information to the chatbot.

# Documents --> Split into Chunks --> Embed into Vector Store --> Conversational Retrieval Chain
# We can use LangChain to create a conversational retrieval chain that can answer questions based on a specific set of documents.
# LangChain is a framework for building applications with LLMs (Large Language Models) like ChatGPT.
# It allows us to create chains of operations that can include loading documents, splitting them into smaller
# chunks, embedding them into a vector store, and then using a conversational retrieval chain to answer questions based on those documents.

heading("Conversational Retrieval Chain")



# We can load data from PDF, text, or web pages using LangChain's document loaders.

