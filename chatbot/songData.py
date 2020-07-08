# used to extract song titles and artist names from large data set
# for training model to identify song requests
import pandas as pd
import re
import numpy as np
import json

# extracting song titles and artist names
data  = pd.read_csv("tracks.txt", sep = "<SEP>", engine="python", header=None)
titles = data[data.columns[len(data.columns)-1]].apply(str)
titles = titles.to_numpy()
for title in titles:
    title = re.sub("[^a-zA-Z]"," ", title)
titles = titles.tolist()
titles = titles[0:999]
 # 999 (chatbotEngine) 9999 (engine)
in_dict = {"tag": "songs",
           "patterns": titles,
           "responses": ["What an awesome choice! I will send your request right over to the host :)", "Marvelous! Your awesome request will be seen by your host", "Scrumptious!! Your amazing request is being sent right away!", "I <3 your request! I hope we will get to hear it very soon", "Your fantastic request is being sent over right now! Feel free to send more requests :)", "What a great taste in music! I will make sure you hear it", "Wowza, I love your request! Your host will see it anytime now :)"],
           "context": [""]
          }

with open('intents.json') as file_object:
    intents = json.load(file_object)
intents["intents"].append(in_dict)

def writetojson(filename, data):
    filename = filename + ".json"
    with open(filename, 'w') as fp:
        json.dump(data, fp)

fileName = "intents_final"
writetojson(fileName, intents)