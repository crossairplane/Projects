import sys
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    '''
    load_data
    Load data from the database we created with clean data.

    Input:
    database_filepath filepath of cleaned database

    Returns:
    X.values messages we use in models
    Y.values the category we need to target
    categories the categories' names
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset', con=engine)


    df = df.dropna()

    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    categories = list(Y)

    return X.values, Y.values, categories


def tokenize(text):
    '''
    Input:
    text: original text messages

    Output:
    tokens: Tokenized, Lemmatize, cleaned data
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(t).lower().strip() for t in tokens]

    return tokens


def build_model():
    '''
    Build Machine Learning model using tf-idf, randomforest and Grid Search

    Output:
    cv : GridSearchCV results
    '''

    pipeline = Pipeline([
        ("vect" , CountVectorizer(tokenizer=tokenize)),
        ("tfidf" , TfidfTransformer()),
        ("clf" , RandomForestClassifier())
    ])

    parameters = {
        'clf__n_estimators': [5, 6, 7],
        'clf__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluate the performance of model.

    Input:
    model: your model
    X_test: Tesing data to input
    Y_test: Tesing data with labels
    category_names: The category names

    Output:
    Accuracy scores for each category
    '''

    predictions = model.predict(X_test)
    print("Accuracy scores for each category\n")
    print("*-" * 30)

    for i in range(36):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], predictions[:, i]))


def save_model(model, model_filepath):
    '''
    save_model
    Save model to a filepath.

    Input:
    model: The name of model
    model_filepath: Filepath to save

    Returns:
    A pickle file.
    '''

    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        predictions = model.predict(X_test)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
