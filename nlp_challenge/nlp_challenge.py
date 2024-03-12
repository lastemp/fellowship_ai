
import os
# import random
import re
import sys
import csv

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

import pandas as pd
# import numpy as np

# TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 3:
        sys.exit("Usage: python nlp_challenge.py aclImdb\\train aclImdb\\test")

    # Load data from spreadsheet and split into train and test sets
    # evidence, labels = load_data_1(sys.argv[1], sys.argv[2])
    # print(len(evidence))
    # print(labels)
    """ X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%") """

    # ***
    """ data = load_data_1(sys.argv[1], sys.argv[2])
    df = pd.DataFrame(data)

    evidence = df["evidence"]
    labels = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(evidence, labels, test_size=0.2, random_state=42) """

    training_data, test_data = load_data(sys.argv[1], sys.argv[2])
    df_training = pd.DataFrame(training_data)
    df_test = pd.DataFrame(test_data)

    X_train = df_training["evidence"]
    y_train = df_training["labels"]

    X_test = df_test["evidence"]
    y_test = df_test["labels"]

    # RandomForestClassifier
    model = Pipeline([("count_vectorizer", CountVectorizer()),
                      ("random_forest", RandomForestClassifier(n_estimators=50, criterion='entropy'))])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    #

    """ # KNeighborsClassifier
    clf = Pipeline([('vectorizer', CountVectorizer()),   
                ('KNN', (KNeighborsClassifier(n_neighbors=10, metric = 'euclidean')))])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # """

    """ # MultinomialNB
    clf = Pipeline([       
     ('vectorizer', CountVectorizer()),   
      ('Multi NB', MultinomialNB())])
    # """

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))


def load_data_1(training_directory, test_directory):
    """
    Load data from text file located in training and test directories.
    """
    data = dict()
    evidence = []
    labels = []
    positive_dir = '\\pos'
    negative_dir = '\\neg'
    positive = 1
    negative = 0

    # training data
    training_positive_dir = training_directory + positive_dir
    training_negative_dir = training_directory + negative_dir

    # test data
    test_positive_dir = test_directory + positive_dir
    test_negative_dir = test_directory + negative_dir

    # training positive data
    if os.path.exists(training_positive_dir):
        # Extract contents in files
        for filename in os.listdir(training_positive_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(training_positive_dir, filename), encoding="utf8") as f:
                contents = f.read()
                evidence.append(contents)  # review
                labels.append(positive)  # positive review

    # training negative data
    if os.path.exists(training_negative_dir):
        # Extract contents in files
        for filename in os.listdir(training_negative_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(training_negative_dir, filename), encoding="utf8") as f:
                contents = f.read()
                evidence.append(contents)  # review
                labels.append(negative)  # negative review

    # test positive data
    if os.path.exists(test_positive_dir):
        # Extract contents in files
        for filename in os.listdir(test_positive_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(test_positive_dir, filename), encoding="utf8") as f:
                contents = f.read()
                evidence.append(contents)  # review
                labels.append(positive)  # positive review

    # test negative data
    if os.path.exists(test_negative_dir):
        # Extract contents in files
        for filename in os.listdir(test_negative_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(test_negative_dir, filename), encoding="utf8") as f:
                contents = f.read()
                evidence.append(contents)  # review
                labels.append(negative)  # negative review

    """ # Extract contents in files
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            evidence.append([contents]) # review
            labels.append(positive)  # positive review """

    # add items to dictionary
    data["evidence"] = evidence
    data["labels"] = labels

    # return evidence, labels
    return data


def load_data(training_directory, test_directory):
    """
    Load data from text file located in training and test directories.
    """
    training_data = dict()
    test_data = dict()
    training_evidence = []
    training_labels = []
    test_evidence = []
    test_labels = []
    positive_dir = '\\pos'
    negative_dir = '\\neg'
    positive = 1
    negative = 0

    # training data
    training_positive_dir = training_directory + positive_dir
    training_negative_dir = training_directory + negative_dir

    # test data
    test_positive_dir = test_directory + positive_dir
    test_negative_dir = test_directory + negative_dir

    # training positive data
    if os.path.exists(training_positive_dir):
        # Extract contents in files
        for filename in os.listdir(training_positive_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(training_positive_dir, filename), encoding="utf8") as f:
                contents = f.read()
                training_evidence.append(contents)  # review
                training_labels.append(positive)  # positive review

    # training negative data
    if os.path.exists(training_negative_dir):
        # Extract contents in files
        for filename in os.listdir(training_negative_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(training_negative_dir, filename), encoding="utf8") as f:
                contents = f.read()
                training_evidence.append(contents)  # review
                training_labels.append(negative)  # negative review

    # test positive data
    if os.path.exists(test_positive_dir):
        # Extract contents in files
        for filename in os.listdir(test_positive_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(test_positive_dir, filename), encoding="utf8") as f:
                contents = f.read()
                test_evidence.append(contents)  # review
                test_labels.append(positive)  # positive review

    # test negative data
    if os.path.exists(test_negative_dir):
        # Extract contents in files
        for filename in os.listdir(test_negative_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(test_negative_dir, filename), encoding="utf8") as f:
                contents = f.read()
                test_evidence.append(contents)  # review
                test_labels.append(negative)  # negative review

    # add items to dictionary
    training_data["evidence"] = training_evidence
    training_data["labels"] = training_labels

    test_data["evidence"] = test_evidence
    test_data["labels"] = test_labels

    return training_data, test_data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


if __name__ == "__main__":
    main()
