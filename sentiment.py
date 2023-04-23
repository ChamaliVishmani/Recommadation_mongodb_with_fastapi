import warnings

warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

import pickle
import re
import gensim
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import pandas as pd

import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag

nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('wordnet')
from nltk.corpus import wordnet

# POS tagger dictionary
pos_dict = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R': wordnet.ADV}

# show all columns
pd.set_option('display.max_columns', None)

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    text = text.strip()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)

    return text


def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist


def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


def process_data():
    # load csv
    df = pd.read_csv("dataset/Food_ordering_feedback_dataset.csv")

    # drop null values
    df = df.dropna()
    # remove duplicates
    df = df.drop_duplicates()

    df['cleaned_text'] = df['value'].apply(lambda x: clean_text(x.lower()))
    df['tokenized_text'] = df['cleaned_text'].apply(lambda x: word_tokenize(x))
    df['POS tagged'] = df['cleaned_text'].apply(token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    # to csv
    df.to_csv("data/sentiment/lemma.csv", index=False)


def train_sentiment():
    # load csv
    df = pd.read_csv("data/sentiment/lemma.csv")

    # drop null values
    df = df.dropna()
    # remove duplicates
    df = df.drop_duplicates()

    # keep only column_name, value
    df = df[['column_name', 'value']]

    # rename to selected_text	sentiment
    df = df.rename(columns={'column_name': 'sentiment', 'value': 'selected_text'})

    print(df.tail())

    df["selected_text"].isnull().sum()
    df["selected_text"].fillna("No content", inplace=True)

    def depure_data(data):
        # Removing URLs with a regular expression
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        data = url_pattern.sub(r'', data)

        # Remove Emails
        data = re.sub('\S*@\S*\s?', '', data)

        # Remove new line characters
        data = re.sub('\s+', ' ', data)

        # Remove distracting single quotes
        data = re.sub("\'", "", data)

        return data

    temp = []
    # Splitting pd.Series to list
    data_to_list = df['selected_text'].values.tolist()
    for i in range(len(data_to_list)):
        temp.append(depure_data(data_to_list[i]))
    list(temp[:5])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(temp))

    print(data_words[:10])

    def detokenize(text):
        return TreebankWordDetokenizer().detokenize(text)

    data = []
    for i in range(len(data_words)):
        data.append(detokenize(data_words[i]))
    print(data[:5])

    data = np.array(data)
    print(data[:5])

    ##### Label encoding
    labels = np.array(df['sentiment'])
    y = []
    for i in range(len(labels)):
        if labels[i] == 'Neutral':
            y.append(0)
        if labels[i] == 'Negative':
            y.append(1)
        if labels[i] == 'Positive':
            y.append(2)
    y = np.array(y)
    labels = tf.keras.utils.to_categorical(y, 3, dtype="float32")
    del y

    #### Data sequencing and splitting
    max_words = 5000
    max_len = 200

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    tweets = pad_sequences(sequences, maxlen=max_len)
    print(tweets)

    print(labels)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, random_state=0)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # save to npy
    np.save('data/sentiment/X_train.npy', X_train)
    np.save('data/sentiment/X_test.npy', X_test)
    np.save('data/sentiment/y_train.npy', y_train)
    np.save('data/sentiment/y_test.npy', y_test)

    # save tokenizer
    with open('data/sentiment/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # model
    model1 = Sequential()
    model1.add(layers.Embedding(max_words, 20))
    model1.add(layers.LSTM(15, dropout=0.5))
    model1.add(layers.Dense(3, activation='softmax'))

    model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Implementing model checkpoins to save the best metric and do not lose it on training.
    checkpoint1 = ModelCheckpoint("data/sentiment/model1_chk/best_model1.hdf5", monitor='val_accuracy', verbose=1,
                                  save_best_only=True, mode='auto',
                                  period=1, save_weights_only=False)
    history = model1.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint1])

    # Bidirectional LTSM model
    model2 = Sequential()
    model2.add(layers.Embedding(max_words, 40, input_length=max_len))
    model2.add(layers.Bidirectional(layers.LSTM(20, dropout=0.6)))
    model2.add(layers.Dense(3, activation='softmax'))
    model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # Implementing model checkpoins to save the best metric and do not lose it on training.
    checkpoint2 = ModelCheckpoint("data/sentiment/model1_chk/best_model2.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                  mode='auto',
                                  period=1, save_weights_only=False)
    history = model2.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint2])


    # save the model
    model1.save('data/sentiment/model/model1.h5')
    model2.save('data/sentiment/model/model2.h5')

    # save model weights
    model1.save_weights('data/sentiment/model_weights/model1_weights.h5')
    model2.save_weights('data/sentiment/model_weights/model2_weights.h5')

    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)

    print("Done Training")

    # Let's load the best model obtained during training
    best_model = keras.models.load_model("data/sentiment/model_weights/best_model2.hdf5")
    best_model.summary()


def predict_sentiment(text):
    # load tokenizer
    with open('data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Let's load the best model obtained during training
    best_model = keras.models.load_model("data/sentiment/model2_chk/best_model2.hdf5")
    best_model.summary()

    max_len = 200
    sentiment = ['Neutral', 'Negative', 'Positive']

    sequence = tokenizer.texts_to_sequences([text])
    test = pad_sequences(sequence, maxlen=max_len)
    print(sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]])

    return sentiment[np.around(best_model.predict(test), decimals=0).argmax(axis=1)[0]]


def pre_process(df):
    # drop null values
    df = df.dropna()

    # drop duplicates
    df = df.drop_duplicates()

    # keep only column_name, value
    df = df[['column_name', 'value']]

    # rename to selected_text	sentiment
    df = df.rename(columns={'column_name': 'sentiment', 'value': 'selected_text'})

    print(df.tail())

    df["selected_text"].isnull().sum()
    df["selected_text"].fillna("No content", inplace=True)

    return df


def depure_data(data):
    # Removing URLs with a regular expression
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)

    # Remove Emails
    data = re.sub('\S*@\S*\s?', '', data)

    # Remove new line characters
    data = re.sub('\s+', ' ', data)

    # Remove distracting single quotes
    data = re.sub("\'", "", data)

    return data


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


def detokenize(text):
    return TreebankWordDetokenizer().detokenize(text)


if __name__ == '__main__':
    # process_data()
    # train_sentiment()
    predict_sentiment('The restaurant provided updates on the status of our order, which was very helpful.')
