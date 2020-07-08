# BopR - the ultimate chatbot for handling song requests
# powered by NLP deep learning model sanga sanga
import numpy as np
import pandas as pd
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
import random
import json
# import the spotify api class
from spotifyAPI import SpotifyAPI
# client ids for the spotify api and the object
client_id = "35d7fb31230a44dea10ce665bb8571c6"
client_secret = "shh its a secret"
spotify = SpotifyAPI(client_id, client_secret)
# load the saved model
model = tf.keras.models.load_model("chatbotEngine")
# functions that will preprocess user inputs
# clean function will tokenize and lemmantize
# bag of words function will turn clean words to bag of words 

stemmer = LancasterStemmer()
# load the json file (possible inputs and outputs for those inputs) with intents into variable
with open('intents_final.json') as file_object:
    intents = json.load(file_object)

# list of total words
words = []
# list for each type of label
phrases = []
# list of the different labels
labels = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize to get each specific word and append to words list
        tokenized = word_tokenize(pattern)
        words.extend(tokenized)
        phrases.append((tokenized, intent['tag']))
    # list for the labels corresponding to the words
    if intent['tag'] not in labels:
        labels.append(intent['tag'])
# convert all words to lowercase while getting it to its stemmed form
words = [stemmer.stem(w.lower()) for w in words if w != '?']
# remove duplicates in the word list and sort
words = sorted(list(set(words)))
# sort the labels
labels = sorted(list(set(labels)))

def cleaner(input):
    input = word_tokenize(input)
    cleaned_input = [stemmer.stem(word.lower()) for word in input]
    return cleaned_input
def bag_of_words(input, words):
    clean_words = cleaner(input)
    bag = [0] * len(words)
    for word in clean_words:
        for x, w in enumerate(words):
            if w == word:
                bag[x] = 1
    return(np.array(bag))

# accessing spotify api to retrieve a json data object for the selected song
# selects the song on the 
def songData(song_name):
    data = spotify.search(query = song_name, search_type="track")
    data = data["tracks"]
    
    if data['total'] == 0:
        return {}
    else:
        for item in data["items"]:
            return item
            break

# chatbot function to return intent of a word or sentence
def chatbot():
    print('Start talking... Enter quit to exit chat')
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break
        
        song_name = sentence
        sentence = pd.DataFrame([bag_of_words(sentence, words)], dtype=float, index=['input'])
        final_out = model.predict([sentence])[0]
        result_index = np.argmax(final_out)
        tag = labels[result_index]
        
        for rp in intents['intents']:
            if rp['tag'] == tag:
                responses = rp['responses']
        print(random.choice(responses))
        # now for accessing spotify api if user inputs a song
        if tag == "songs":
            out_data = songData(song_name)
            if out_data == {}:
                print("OOPS! While sending your request, I sadly couldn't find your requested song :(")
                print("Please try entering your song title again with the artist name or try another request!")
            else:
                return out_data

print(chatbot())
