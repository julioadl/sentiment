#!/bin/sh
#python3.6 run_experiment.py --gpu=-1 '{"dataset": "Sentiment140CharacterLevelSmall", "model": "Sentiment140Model", "algorithm": "lstm_cnn_chars", "algorithm_args": {"dropout": 0.75}, "train_args": {"epochs": 100, "lr":	0.002}}'
python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevelSmall", "model": "Sentiment140Model", "algorithm": "lstm_chars", "algorithm_args": {"dropout": 0.5}, "train_args": {"epochs": 100, "lr":	0.001, "beta_1": 0.999}}'
#python3.6 run_experiment.py --gpu=-1 '{"dataset": "Sentiment140CharacterLevelSmall", "model": "Sentiment140Model", "algorithm": "cnn_chars", "algorithm_args": {"dropout": 0}, "train_args": {"epochs": 50, "lr":	0.001}}'
