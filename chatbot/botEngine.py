# BopR bot engine for JukeR - Sangaman Senthil
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

# initialize the stemmer
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
# initialize training set and its labels for bag of words
training = []
output_zero = [0] * len(labels)
# creating bag of words (if the word is in phrases list it has 1 if it is not it is 0)
for phrase in phrases:
    # intialize bag of words
    bag = []
    # list of tokenized
    pattern_words = phrase[0]
    # stem each word in pattern words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # creating the bag of words array 1 means it has 0 means it does not
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_zero)
    output_row[labels.index(phrase[1])] = 1
    training.append([bag, output_row])
# shuffle and turn into array for training
random.shuffle(training)
training = np.array(training)
# create train and test lists; x - patterns, y -intents
train_x = list(training[:,0])
train_y = list(training[:,1])

# creating the model
# 3 layers, the third being output layer
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# compile the model with stochastic gradient descent
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# fit the model with training data 
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# save the model to avoid compiling the model each time the chatbot runs
model.save("chatbotEngine")