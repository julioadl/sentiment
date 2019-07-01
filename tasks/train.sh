#!/bin/sh
#python3.6 run_experiment.py --gpu=-1 '{"dataset": "Sentiment140CharacterLevel", "model": "Sentiment140Model", "algorithm": "lstm_cnn_chars", "algorithm_args": {"dropout": 0.25}, "train_args": {"epochs": 100, "lr":	0.01}}'
python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevelLM", "model": "Sentiment140LMModel", "algorithm": "lstm_chars_simple", "algorithm_args": {"dropout": 0.5}, "train_args": {"epochs": 10000, "lr":	0.001}}'
