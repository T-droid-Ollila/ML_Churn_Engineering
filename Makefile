# Remembering lines to style our code is a bit cumbersome
# so it's a good idea to create a Makefile.
# This file can be used to define a set of commands that can be executed with a single command

# Makefile
DATASET_LOC := churn_data/train_churn.csv

ifeq ($(OS), WINDOWS_NT)
# Styling
# PHINY prevents Make from confusing a target name with a real file
.PHONY: style
style:
    black .
    flake8
    isort .

.PHONY: test
test:
    pytest --cov=scripts tests/

.PHONY: test-data
test-data:
	pytest --dataset-loc=$(DATASET_LOC) tests/test_dataset.py

.PHONY: train
train: scripts\train.py --model random_forest --mode search
