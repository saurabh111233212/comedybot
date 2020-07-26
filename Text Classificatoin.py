import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

print(train_data[0])

word_index = data.get_word_index()

# maps words to integers
word_index = {k: (v+3) for (k, v) in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# a little backMap to map the integers to words (for display purposes)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# pre-processes the reviews to ensure they're all the same length (250)
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


# takes an array of ints and converts it to the string representing the review (for display purposes)
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# this is for training a new model!
'''
# build the model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(omptimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = test_data[:10000]
x_train = train_data[10000:]

y_val = test_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512,  validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

model.save("text_clf_model.h5") '''

def review_encode(s):
    encoded = [1]
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


model = keras.models.load_model("text_clf_model.h5")

# process the review.txt file and send it to our model
with open("review.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])