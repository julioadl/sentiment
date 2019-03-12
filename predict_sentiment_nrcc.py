from typing import List, Tuple, Union

import numpy as np
import json

from datasets.base import Dataset

PROCESSED_ESSENTIALS_PATH = Dataset.data_dirname() / 'processed' / 'essentials'

class predictor:
    def __init__(self):
        self.nrcc = json.load(open(PROCESSED_ESSENTIALS_PATH / 'sentiment.json'))

    #Designed to take a list of tuples. In each tuple, the first element is the ID
    #The second is the sentence
    #Returns a list of lists - each list witthin pred contains the ID, the sentence
    # and the score for each of the sentiments within NRCC
    def predict(self, text: List) -> Tuple[str, float]:
        pred = []
        for sentence in text:
            stc = sentence.split()
            sentiment = {'anger': 0,
            'anticipation': 0,
            'disgust': 0,
            'fear': 0,
            'joy': 0,
            'negative': 0,
            'positive': 0,
            'sadness': 0,
            'surprise': 0,
            'trust': 0}

            for word in stc:
                try:
                    sentiment['anger'] += int(self.nrcc[word]['anger'])
                    sentiment['anticipation'] += int(self.nrcc[word]['anticipation'])
                    sentiment['disgust'] += int(self.nrcc[word]['disgust'])
                    sentiment['fear'] += int(self.nrcc[word]['fear'])
                    sentiment['joy'] += int(self.nrcc[word]['joy'])
                    sentiment['negative'] += int(self.nrcc[word]['negative'])
                    sentiment['positive'] += int(self.nrcc[word]['positive'])
                    sentiment['sadness'] += int(self.nrcc[word]['sadness'])
                    sentiment['surprise'] += int(self.nrcc[word]['surprise'])
                    sentiment['trust'] += int(self.nrcc[word]['trust'])
                except:
                    pass
            pred.append([sentence, sentiment['anger'], sentiment['anticipation'],
                        sentiment['disgust'], sentiment['fear'], sentiment['joy'],
                        sentiment['negative'], sentiment['positive'], sentiment['sadness'],
                        sentiment['surprise'], sentiment['trust']])
        return pred
