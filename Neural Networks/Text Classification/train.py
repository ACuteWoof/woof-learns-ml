import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

num_words = 20000

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=num_words)

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# MODEL

model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:num_words]
x_train = train_data[num_words:]

y_val = train_labels[:num_words]
y_train = train_labels[num_words:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

print(model.predict(test_data))
results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")
