
import os
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import pandas as pd


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python nlp_challenge.py aclImdb")

    # Load data from text files(train/test sub-directories) and allocate into train and test sets
    train_data, test_data = load_data(sys.argv[1])
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    X_train = df_train["evidence"]
    y_train = df_train["labels"]

    X_test = df_test["evidence"]
    y_test = df_test["labels"]

    # Train model(RandomForestClassifier) and make predictions
    model = train_model_random_forest_classifier(X_train, y_train)
    y_predictions = model.predict(X_test)

    """ # Train model(KNeighborsClassifier) and make predictions
    model = train_model_k_neighbors_classifier(X_train, y_train)
    y_predictions = model.predict(X_test) """

    """ # Train model(MultinomialNB) and make predictions
    model = train_model_multinomial_nb(X_train, y_train)
    y_predictions = model.predict(X_test) """

    print(classification_report(y_test, y_predictions))


def load_data(directory):
    """
    Load data from text files located in train and test directories.
    """
    train_data = dict()
    test_data = dict()
    train_evidence = []
    train_labels = []
    test_evidence = []
    test_labels = []
    train_directory = '\\train'
    test_directory = '\\test'
    positive_dir = '\\pos'
    negative_dir = '\\neg'
    positive = 1
    negative = 0

    # train data
    train_positive_dir = directory + train_directory + positive_dir
    train_negative_dir = directory + train_directory + negative_dir

    # test data
    test_positive_dir = directory + test_directory + positive_dir
    test_negative_dir = directory + test_directory + negative_dir

    # train positive data
    if os.path.exists(train_positive_dir):
        # Extract contents in files
        for filename in os.listdir(train_positive_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(train_positive_dir, filename), encoding="utf8") as f:
                contents = f.read()
                train_evidence.append(contents)  # review
                train_labels.append(positive)  # positive review

    # train negative data
    if os.path.exists(train_negative_dir):
        # Extract contents in files
        for filename in os.listdir(train_negative_dir):
            if not filename.endswith(".txt"):
                continue
            with open(os.path.join(train_negative_dir, filename), encoding="utf8") as f:
                contents = f.read()
                train_evidence.append(contents)  # review
                train_labels.append(negative)  # negative review

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
    train_data["evidence"] = train_evidence
    train_data["labels"] = train_labels

    test_data["evidence"] = test_evidence
    test_data["labels"] = test_labels

    return train_data, test_data


def train_model_random_forest_classifier(evidence, labels):
    """
    Given a list of evidence and a list of labels, return a
    fitted RandomForestClassifier model trained on the data.
    """
    model = Pipeline([("count_vectorizer", CountVectorizer()),
                      ("random_forest", RandomForestClassifier(n_estimators=50, criterion='entropy'))])
    model.fit(evidence, labels)
    return model


def train_model_k_neighbors_classifier(evidence, labels):
    """
    Given a list of evidence and a list of labels, return a
    fitted k-nearest neighbor model (k=10) trained on the data.
    """
    model = Pipeline([('vectorizer', CountVectorizer()),
                      ('KNN', (KNeighborsClassifier(n_neighbors=10, metric='euclidean')))])
    model.fit(evidence, labels)
    return model


def train_model_multinomial_nb(evidence, labels):
    """
    Given a list of evidence and a list of labels, return a
    fitted MultinomialNB model trained on the data.
    """
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('Multi NB', MultinomialNB())])
    model.fit(evidence, labels)
    return model


if __name__ == "__main__":
    main()
