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

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
)
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
)

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

tweets_df = create_tweets_df(["depression"])



#SPLITTING THE TRAINING DATASET INTO TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(tweets_df["tweet_text"], tweets_df["has_disorder"], test_size=TEST_RATIO, shuffle=True)

#Word2Vec (TODO: get this working later?)
# Word2Vec runs on tokenized sentences
X_train_tok = [tweet_tokenizer.tokenize(i) for i in X_train]  
X_test_tok = [tweet_tokenizer.tokenize(i) for i in X_test]

# Term Frequency-Inverse Document Frequencies
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_test_vectors_tfidf = tfidf_vectorizer.transform(X_test)

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

tweets_df["tweet_text_tok"] = [tweet_tokenizer.tokenize(i) for i in tweets_df["tweet_text"]]
model = Word2Vec(tweets_df["tweet_text_tok"], min_count=1)
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) # TODO: clean up this copy-pasted code
modelw = MeanEmbeddingVectorizer(w2v)

# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_test_vectors_w2v = modelw.transform(X_test_tok)



#FITTING THE CLASSIFICATION MODEL using Logistic Regression(tf-idf)
print("\033[32mFitting LogisticRegression classification model using tf-idf for vectorization...\033[0m")
lr_tfidf = LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_tfidf.fit(X_train_vectors_tfidf, y_train)  

#Predict y value for test dataset
y_predict = lr_tfidf.predict(X_test_vectors_tfidf)
y_prob = lr_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
print(classification_report(y_test, y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
print()



#FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
print("\033[32mFitting LogisticRegression classification model using w2v for vectorization...\033[0m")
lr_w2v=LogisticRegression(solver='liblinear', C=10, penalty='l2')
lr_w2v.fit(X_train_vectors_w2v, y_train)
#Predict y value for test dataset
y_predict = lr_w2v.predict(X_test_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_test_vectors_w2v)[:,1]
print(classification_report(y_test, y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
print()



#FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
print("\033[32mFitting MultinomialNB classification model using tf-idf for vectorization...\033[0m")
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_vectors_tfidf, y_train)  

#Predict y value for test dataset
y_predict = nb_tfidf.predict(X_test_vectors_tfidf)
y_prob = nb_tfidf.predict_proba(X_test_vectors_tfidf)[:,1]
print(classification_report(y_test, y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
print()
