from typing import List, Dict, Tuple

import numpy as np
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

'''
Creates a np.array with the data coming from fakenews.py
Takes a list of the following values:
    text: str

Returns numpy array, numpy array, dict of features
'''
def vectorizer(text: List[str], languages: List[str] = ["english"],
                id_start: int=1, max_length_str: int=240, features: Dict=None,
                labels: List[str] = None):

    if features is None:
        features = _generateIndexes(text)
    X = _generateIndexArray(text, features, max_length_str=max_length_str)

    #Process labels if necessary
    if labels is not None:
        categories = preprocessing.LabelEncoder()
        Y = categories.fit_transform(labels)
    else:
        Y = None

    return X, Y, features

def _generateIndexes(text: List['str']):
    vocabulary = []
    for sentence in text:
        for char in sentence:
            vocabulary.extend(char)
    vocabulary = list(set(vocabulary))
    nVoc = len(vocabulary)
    indices = range(1, nVoc + 1)
    vocabularyDict = dict(zip(vocabulary, indices))
    inverseVocabularyDict = dict(zip(indices, vocabulary))
    return vocabularyDict

def _generateIndexArray(text: List['str'], features: Dict, max_length_str: int=120):
    noRows = len(text)
    X = np.zeros((noRows, max_length_str))
    for i in range(noRows):
        sentence = text[i]
        j = 0
        for char in sentence:
            if j==max_length_str:
                break
            try:
                X[i,j] = features[char]
            except:
                pass
            j += 1
    return X
