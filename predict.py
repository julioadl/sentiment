from typing import List, Tuple, Union

import numpy as np
import json

from models import Sentiment140Model
from utils.preprocess_tokenize_by_char import vectorizer
from datasets.base import Dataset

class predictor:
    def __init__(self):
        self.model = Sentiment140Model()
        self.model.load_weights()
        self.features = self.model.load_features()

    def predict(self, text: List[str]) -> Tuple[str, float, List]:
        textMatrix, _, _ = vectorizer(text, self.features)
        pred = self.model.predict(textMatrix)
        return pred
