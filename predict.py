from typing import List, Tuple, Union, Optional

import numpy as np
import random
import json

from models import Sentiment140LMModel
from utils.preprocess_tokenize_by_char import vectorizer
from datasets.base import Dataset

PROCESSED_ESSENTIALS_PATH = Dataset.data_dirname() / 'processed' / 'essentials'
STRING_FILE = PROCESSED_ESSENTIALS_PATH / 'string_character_level_lm.json'
CHARACTER_INDEX = PROCESSED_ESSENTIALS_PATH / 'features_character_level_lm.json'
INDICES_CHARACTER = PROCESSED_ESSENTIALS_PATH / 'indices_char_character_level_lm.json'

char_index = json.load(open(CHARACTER_INDEX))
index_char = json.load(open(INDICES_CHARACTER))
text = json.load(open(STRING_FILE))

class predictor:
    def __init__(self):
        self.model = Sentiment140LMModel(algorithm_args={"num_layers": 3, "hidden_units": 128, "dropout": 0})
        self.model.load_weights()
        #self.features = self.model.load_features()

    def predict(self, maxlen: Optional[int] = 40) -> str:
        for diversity in [0.3, 0.4, 0.5, 0.6]:
            print('----- diversity:', diversity)
            generated = ''
            sentence = ' ' * (maxlen - 2) + '≥≤'
            generated += sentence
            #print('----- Generating with seed: "' + sentence + '"')
            #print(generated)

            for i in range(140):
                x_pred = np.zeros((1, maxlen))
                for t, char in enumerate(sentence):
                    x_pred[0, t] = char_index[char]

                preds, _, probas = self.model.predict(x_pred)
                next_index = self.sample(probas, diversity)
                next_char = index_char[str(next_index)]

                sentence = sentence[1:] + next_char

            print(sentence)
#        return sentence

    def sample(self, preds, temperature=1.0):
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
