# ENGG*6600 ST: Deep Learning Term Project W24
# Mental Illness Classification with RNNs
# Dataset: https://zenodo.org/records/6409736
# 
# Ben Chapman-Kish (bchapm02@uoguelph.ca)
# John Quinto (jquinto@uoguelph.ca)
# Om Bhosale (obhosale@uoguelph.ca)
# Parya Abadeh (pabadeh@uoguelph.ca)
# 
# Basic proof-of-concept network to perform the disorder classification task
# This is basically all just ripped from https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e

from preprocessing import create_tweets_df, tweet_tokenizer
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Masking, LSTM, Dense, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight

import numpy as np

from prepare_sequences import prepare_sequences

from sklearn.model_selection import train_test_split

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
from gensim.models import Word2Vec

TEST_RATIO = 0.2

tweets_df = create_tweets_df(["depression", "anxiety"])
tweets_df["tweet_text_tok"] = [tweet_tokenizer.tokenize(i) for i in tweets_df["tweet_text"]]

def create_word_embeddings(word2vec, tweets_tok):
  modelw = MeanEmbeddingVectorizer(word2vec)
  return modelw.transform(tweets_tok)



#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    


model = Word2Vec(tweets_df["tweet_text_tok"], min_count=1)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))

# Generate embeddings for all tweets
tweet_embeddings = create_word_embeddings(w2v, tweets_df["tweet_text_tok"])

tweet_embeddings_list = tweet_embeddings.tolist()
tweets_df["tweet_embedding"] = tweet_embeddings_list

sequences, labels = prepare_sequences(tweets_df.copy())


max_sessions = 100

for i, sequence in enumerate(sequences):
    if len(sequence) < max_sessions:
        padding_array = [0] * 100
        for j in range(0, (max_sessions - len(sequence))):
            sequences[i].append(padding_array)


labels = np.array(labels)
print(labels.shape)
sequences = np.array(sequences)
print(sequences.shape)

early_stopper = EarlyStopping(monitor='val_loss',  
                              patience=10,  
                              verbose=1,
                              mode='min',
                              restore_best_weights=True)  


X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 

X_train = np.array(X_train)
X_test = np.array(X_test)
X_val = np.array(X_val)

batch_size = 32

# Define the LSTM model
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(max_sessions, 100)))
model.add(LSTM(64, return_sequences=True, dropout=0.2)) 
model.add(LSTM(64, dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopper]) 


loss, accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()