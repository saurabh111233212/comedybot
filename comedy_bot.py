import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
import nltk
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# preprocess the data
df = pd.read_csv("jokes.csv")
jokes = df['text']
labels = list(df['humor'])
# lemmatize each word in the jokes
lemmatizer = WordNetLemmatizer()
jokes = [np.array(joke.split(' ')) for joke in jokes]
for joke in jokes:
    for i in range(len(joke)):
        joke[i] = lemmatizer.lemmatize(joke[i])

# convert everything to numbers (hashing)
for i in range(len(labels)):
    labels[i] = 1 if labels[i] else 0
labels = np.array(labels)

for joke in jokes:
    for i in range(len(joke)):
        joke[i] = hash(joke[i])
jokes = np.array(jokes)

all_words = np.concatenate(jokes.flatten())
unique_words = np.unique(all_words)

# build the neural network
model = keras.Sequential()
model.add(keras.layers.Embedding(len(unique_words) + 1, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss="binary_crossentropy", metrics=["acc"])

training_jokes, testing_jokes, training_labels, testing_labels = sklearn.model_selection.train_test_split(jokes,
                                                                                                          labels,
                                                                                                          test_size=.2)
model.fit(training_jokes, training_labels)
results = model.evaluate(testing_jokes, testing_labels)

model.save("comedy_model.h5")

