import streamlit as st
import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Initialiser le lemmatizer
lem = WordNetLemmatizer()

# Charger les données d'intentions
with open('intents.json') as f:
    data = json.load(f)

# Prétraitement des données
classes = []  # liste des différents labels
words = []  # liste des différents mots
documentX = []  # liste des différents patterns
documentY = []  # liste des labels de chaque pattern

# Tokeniser les patterns et rajouter les questions dans documentX et les étiquettes dans documentY
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tkns = nltk.word_tokenize(pattern)
        words.extend(tkns)
        documentX.append(pattern)
        documentY.append(intent['tag'])
    if intent["tag"] not in classes:
        classes.append(intent['tag'])

words = [lem.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

# Vectoriser les phrases de nos patterns
vectorizer = CountVectorizer(vocabulary=words)
sentences_object = vectorizer.fit_transform(documentX)
x = sentences_object.toarray()

# Encodage des étiquettes
y = np.array(pd.get_dummies(documentY))

# Créer le modèle
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(x, y, epochs=200)

# Fonction de prédiction
def tk_lm_func(text):
    tkns = nltk.word_tokenize(text)
    tkns = [lem.lemmatize(word) for word in tkns]
    return tkns

def vectorizer_func(text, vocab):
    tkns = tk_lm_func(text)
    sent_vec = [0] * len(vocab)
    for w in tkns:
        for idx, word in enumerate(vocab):
            if word == w:
                sent_vec[idx] = 1
    return np.array(sent_vec)

def Pred_func(text, vocab, labels):
    sent_vec = vectorizer_func(text, vocab)
    result = model.predict(np.array([sent_vec]))
    result = result.argmax(axis=1)
    tag = labels[result[0]]
    return tag

def get_res(tag, fJson):
    list_intents = fJson['intents']
    for i in list_intents:
        if i["tag"] == tag:
            ourResult = random.choice(i['responses'])
            break
    return ourResult

# Interface utilisateur avec Streamlit
st.title("Chatbot avec Streamlit")
user_input = st.text_input("Vous: ", "Bonjour")
if user_input:
    tag = Pred_func(user_input, words, classes)
    response = get_res(tag, data)
    st.text_area("Chatbot:", response, height=200)
