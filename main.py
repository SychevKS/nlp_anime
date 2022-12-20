import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from clean_text import clean_text

df = pd.read_excel("./anime.xlsx")
df["text"] = [clean_text(t) for t in df["text"]]


tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(df["text"])
texts = tokenizer.texts_to_sequences(df["text"])
labels = np.array(df["label"])

vectorizer = CountVectorizer(binary=True, analyzer=lambda x: x,  max_features=10000)
texts = vectorizer.fit_transform(texts).toarray()
print(texts[0], len(texts[0]))

x_train, x_test, y_train, y_test = train_test_split(
    texts, 
    labels, 
    test_size=0.3,
    stratify=labels, 
    random_state=42)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(texts[0]),)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

plt.plot(history.history['accuracy'], label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_accuracy'], label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

scores = model.evaluate(x_test, y_test, verbose=1)
print("Доля верных ответов на тестовых данных, в процентах:", round(scores[1] * 100, 2))

def predict(text):
    text = tokenizer.texts_to_sequences(text)
    text = vectorizer.fit_transform(texts).toarray()
    text =  pad_sequences(text, maxlen=len(texts[0]), padding='post')
    result = model.predict(text)
    return result[0]

while(True):
    desc = input('Описание: ')
    print(predict(desc))
