#!/bin/sh
#python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevelLM", "model": "Sentiment140LMModel", "algorithm": "lstm_cnn_chars", "algorithm_args": {"dropout": 0.5}, "train_args": {"epochs": 50, "lr":	0.0001, "beta_1": 0.999}}'
python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevelLM", "model": "Sentiment140LMModel", "algorithm": "lstm_cnn_chars", "algorithm_args": {"num_layers": 3, "hidden_units": 128, "dropout": 0.5}, "train_args": {"epochs": 10, "lr":	0.0001}}'
#python3.6 run_experiment.py --gpu=-1 --save '{"dataset": "Sentiment140CharacterLevelLM", "model": "Sentiment140LMModel", "algorithm": "lstm_chars", "algorithm_args": {"dropout": 0}, "train_args": {"epochs": 100, "lr":	0.001}}'
