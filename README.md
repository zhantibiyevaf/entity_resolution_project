# record_matching_project
This repository implements a simple record matching pipeline which is designed to identify whether two records refer to the same person across the datasets. 

## classifiers/match_classifier.py
This file contains the reusable classifier which is used to train and predict whether two records match.

## features/match_features.py
This code creates similarity features between pairs of records such as name similarity, affiliation, and distance.

## readers/data_pairs.py
Loads data and generated candidate record pairs that will later be evaluated by the model.

## scripts/train_test_matching.py
This code runs the full pipeline including feature creation, training the model, and predicting it.

## Goal of the project
The goal of this project is to identify dublicate or matching records in datasets without comparing every pair manually.

