'''
Sentiment 140 dataset
For further reference see:
    http://help.sentiment140.com/for-students
'''
import json
import io
import os
import pathlib
import zipfile
from urllib.request import urlretrieve

from boltons.cacheutils import cachedproperty
import h5py
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.externals import joblib
from tensorflow.keras.utils import to_categorical
from collections import Counter

from .base import Dataset
from utils.preprocess_tokenize_by_char import vectorizer

#ADD URL
#http://help.sentiment140.com/for-students
RAW_URL = 'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip'
NRCC_URL = 'http://sentiment.nrc.ca/lexicons-for-research/NRC-Sentiment-Emotion-Lexicons.zip'

RAW_DATA_DIRNAME = Dataset.data_dirname() / 'raw' / 'sentiment140'
PROCESSED_DATA_DIRNAME = Dataset.data_dirname() / 'processed' / 'sentiment140'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / 'data_sentiment140_character_level.h5'
RAW_ESSENTIALS_PATH = Dataset.data_dirname() / 'raw' / 'essentials'
PROCESSED_ESSENTIALS_PATH = Dataset.data_dirname() / 'processed' / 'essentials'
ESSENTIALS_DATA_FILENAME = PROCESSED_ESSENTIALS_PATH / 'features_character_level.json'

def _download_data():

    RAW_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    RAW_ESSENTIALS_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_ESSENTIALS_PATH.mkdir(parents=True, exist_ok=True)

    os.chdir(RAW_DATA_DIRNAME)

    if not os.path.exists('sentiment140.zip'):
        print('Downloading data...')
        urlretrieve(RAW_URL, 'sentiment140.zip')
        zip_file = zipfile.ZipFile('sentiment140.zip', 'r')
        zip_file.extractall()

    os.chdir(RAW_ESSENTIALS_PATH)

    if not os.path.exists('nrcc.zip'):
        print('Downloading NRCC  Lexicons...')
        urlretrieve(NRCC_URL, 'nrcc.zip')
        zip_file = zipfile.ZipFile('nrcc.zip', 'r')
        zip_file.extract('NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', 'nrcc')
        fname = 'nrcc/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
        sentiment_dict = _process_nrcc(fname)
        with open(PROCESSED_ESSENTIALS_PATH / 'sentiment.json', 'w') as f:
            json.dump(sentiment_dict, f)

    print('Data downloaded and pre-processed...')

def _process_data():

    print('Processing data ...')

    #os.chdir(PROCESSED_DATA_DIRNAME)
    if not os.path.exists(PROCESSED_DATA_FILENAME):
        #Training data
        print('Creating character_level data...')
        #Read files, take columns and pass them as a list
        #Appends only the first 300K values because of memory issues... append all the others later
        df = pd.read_csv(RAW_DATA_DIRNAME / 'training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")
        df = df.dropna()
        df.columns = ['polarity', 'id', 'date', 'query', 'user_screen_name', 'text']
        df = df.drop(columns=['id', 'date', 'query', 'user_screen_name'])
        df = df.sample(100000, random_state=42)
        #Drop Nulls
        #Notice these columns come from reading the documentation at:
        #http://help.sentiment140.com/for-students
        labels_train = list(df['polarity'])
        text_train = list(df['text'])
        #Generate index matrix
        X_train, y_train, features = vectorizer(text_train, labels=labels_train)
        y_train = to_categorical(y_train)

        #Test data
        df = pd.read_csv(RAW_DATA_DIRNAME / 'testdata.manual.2009.06.14.csv')
        df.columns = ['polarity', 'id', 'date', 'query', 'user_screen_name', 'text']
        df = df[df.polarity != 2]
        #Drop Nulls
        df = df.dropna()
        #Notice these columns come from reading the documentation at:
        #http://help.sentiment140.com/for-students
        df.columns = ['polarity', 'id', 'date', 'query', 'user_screen_name', 'text']
        df = df.drop(columns=['id', 'date', 'query', 'user_screen_name'])
        labels_test = list(df['polarity'])
        text_test = list(df['text'])

        #Generate index matrix
        X_test, y_test, _ = vectorizer(text_test, features=features, labels=labels_test)
        y_test = to_categorical(y_test)

        print('Saving to HDF5...')
        with h5py.File(PROCESSED_DATA_FILENAME, 'w') as f:
            f.create_dataset('X_train', data=X_train, compression='lzf')
            f.create_dataset('y_train', data=y_train, compression='lzf')
            f.create_dataset('X_test', data=X_test, compression='lzf')
            f.create_dataset('y_test', data=y_test, compression='lzf')

        with open(ESSENTIALS_DATA_FILENAME, 'w') as f:
            json.dump(features, f)

        print('Done')

def _process_nrcc(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = []
    fin.readline()
    for line in fin:
        data.append(line.split())

    data = np.array(data)
    words_list = np.unique(data[:,0])

    word_dict = {}
    for word in words_list:
        word_dict[word] = {}

    for word, feeling, score in data:
        feeling_dict = {}
        feeling_dict[feeling] = score
        word_dict[word].update(feeling_dict)

    return word_dict

class Sentiment140CharacterLevel(Dataset):
    def __init__(self):
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_data()
            _process_data()
        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as f:
            self.x_train = f['X_train'][:]
            self.y_train = f['y_train'][:]
            self.x_test = f['X_test'][:]
            self.y_test = f['y_test'][:]

        self.features = json.load(open(ESSENTIALS_DATA_FILENAME))

        self.input_shape = self.x_train.shape[1:]
        self.output_shape = self.y_train.shape[1:]
        self.mapping = {0: 'negative', 1: 'positive'}

    def __repr__(self):
        return (
            'Sentiment140 character level\n'
            '3 classes: Negative, Neutral, Positive\n'
            f'Input shape: {self.input_shape}\n'
            f'Output shape: {self.output_shape}'
        )

if __name__=='__main__':
    data = Sentiment140CharacterLevel()
    print(data.__repr__)
    print(data.x_train)
