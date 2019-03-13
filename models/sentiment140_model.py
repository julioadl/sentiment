from typing import Callable, Dict, Tuple
import numpy as np
import pathlib
import tensorflow
from tensorflow.keras.models import load_model

from .base import ModelTf as Model
from datasets.sentiment140_character_level import Sentiment140CharacterLevel
from algorithms.lstm_chars import lstm_chars

DIRNAME = pathlib.Path('__file__').parents[0].resolve() / 'weights'

class Sentiment140Model(Model):
    def __init__(self, dataset_cls: type=Sentiment140CharacterLevel, algorithm_fn: Callable=lstm_chars, dataset_args: Dict=None, algorithm_args: Dict=None):
        super().__init__(dataset_cls, algorithm_fn, dataset_args, algorithm_args)

    def load_features(self):
        features = self.data.features
        return features

    def predict(self, text_as_embedding: np.ndarray) -> Tuple[str, float]:
        pred_raw = self.algorithm.predict(text_as_embedding, batch_size=1).flatten()
        ind = np.argmax(pred_raw)
        confidence_of_prediction = pred_raw[ind]
        predicted_character = self.data.mapping[ind]
        return (predicted_character, confidence_of_prediction, pred_raw)
