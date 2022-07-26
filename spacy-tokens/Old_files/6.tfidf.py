# Coding in utf-8
"""
Created on date: 19 July 2022
@authors: Timothy J. Naudet

Purpose: Perform tfidf vectorization on the tokens collcted by the tokenizers

"""

import argparse
import requests
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from _utilities import split_data, clean_dataframe, git_raw_urls, request_url_data, write_json, array_writing


# Function to import the dataset
def collect_data(json_file):
    df = pd.DataFrame(columns=['text'])
    for _dict_ in json_file:
        df.loc[_dict_['id']] = _dict_['spacy']
    return df


# function for topic modeling
def topic_modeling(df):
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = tfidf.fit_transform(df['text'])
    nmf_model = NMF(n_components=5, random_state=42)
    nmf_model.fit(dtm)
    df['Topic #'] = nmf_model.transform(dtm).argmax(axis=1)
    my_topic_dictionary = {1:'health', 2:'topic2', 3:'election', 4:'poli', 5:'election'}
    df['Topic Label'] = df['Topic #'].map(my_topic_dictionary)
    return df


# Function to perform TFIDF Pipelining
def tfidf_clusering(X_train, y_train, X_test):
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)
    predictions = text_clf.predict(X_test)
    return predictions


# function to assign labels to the data for their file source
def assign_labels(df):
    labels = []
    for long_text in df['text']:
        for word in long_text.split(' '):
            if re.search(r'[^./][\w+][/\w+]', word) and len(word) > 6:
                if 'loki' in word:
                    labels.append('loki')
                elif 'ldconfig' in word:
                    labels.append('ldconfig')
                else:
                    labels.append('other')
            else:
                labels.append('None')
    # the below dimensions must be the same
    df['labels'] = labels
    return df


# Function to perform metrics
def metrics(y_test, predictions):
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    return -1

# Function to plot the clusters
def plotting():
    pass

# Main function
def main():
    """
    Need to read in each error-#.json file, add each set of tokens as a separate string into a corpus, and vectorize it.

    Another issue is that the data is essentially without labels; we can use topic modeling to stop this?

    :return: nothing
    """
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repository name')
    args = parser.parse_args()
    user_name = args.User_name
    repository = args.Repository
    raw_urls = git_raw_urls(user_name, repository, "spacy")
    index = 0

    # if using tfidf, this should become a dataframe where the text is a column
    token_corpus = pd.DataFrame()

    '''Section for tfidf clustering'''
    # for url in raw_urls[0:1]:
    #     requested_json = requests.get(url).json()
    #     df = collect_data(requested_json)
    #     df = assign_labels(df=df)
    #
    #     X_train, X_test, y_train, y_test = split_data(df['text'], df['labels'], 0.33, 42)
    #
    #     predictions = tfidf_clusering(X_train, y_train, X_test)
    #     metrics(y_test=y_test, predictions=predictions)


    '''Section for Topic Modeling'''
    for url in raw_urls[0:1]:
        requested_json = requests.get(url).json()
        df = collect_data(requested_json)
        df = topic_modeling(df)
        array_writing(array=np.array(df), header="This is the header", filename="_test_array_for_topmod.txt")


if __name__ == '__main__':
    main()

    # Configurations
    # python 6.tfidf TimJN17 spack-monitor-nlp