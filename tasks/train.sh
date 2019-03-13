#!/bin/sh
#python3.6 run_experiment.py --gpu=-1 '{"dataset": "Sentiment140CharacterLevel", "model": "Sentiment140Model", "algorithm": "lstm_cnn_chars", "algorithm_args": {"dropout": 0.25}, "train_args": {"epochs": 100, "lr":	0.01}}'
python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevel", "model": "Sentiment140Model", "algorithm": "lstm_chars", "algorithm_args": {"dropout": 0.25}, "train_args": {"epochs": 1, "lr":	0.01}}'
