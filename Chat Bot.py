import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import pickle
import os

with open("intents.json") as file:
    data = json.load(file)

try:
    # try loading the lists if they exist
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # stem the words in the patterns
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            curr_pattern = nltk.word_tokenize(pattern)
            words.extend(curr_pattern)
            docs_x.append(curr_pattern)
            tag = intent["tag"]
            docs_y.append(tag)
            if tag not in labels:
                labels.append(tag)

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # create bags of words, 1 hot encoded
    for x, doc in enumerate(docs_x):
        bag = []
        curr_pattern = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            if w in curr_pattern:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    # save the lists
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tf.reset_default_graph()

# building the neural network's layers (building the model itself)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# load the model if it exists, otherwise go ahead and train
if os.path.exists("chatBot_model" + ".meta"):
    model.load("chatBot_model")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("chatBot_model")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for sen_word in s_words:
        for i, w in enumerate(words):
            if w == sen_word:
                bag[i] = 1
    return np.array(bag)


def chat():
    print("Start chatting with the bot. Type \"quit\" to stop")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        res = model.predict(([bag_of_words(inp, words)]))[0]
        res_idx = np.argmax(res)
        tag = labels[res_idx]

        # need a confidence of 70% to display a response
        if res[res_idx] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]

            print(random.choice(responses))
        else:
            print("Not sure how to respond. Try again?")
chat()

