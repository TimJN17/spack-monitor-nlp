# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet...

Purpose: Plot the most common 30 tokens and the most common 30 file paths

"""

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import re
import matplotlib.pyplot as plt
from collections import Counter
from _utilities import read_json
from string import punctuation


# Function for plotting
def plot_tokens(incoming_json, counter, file_path_ner):
    sns.set_theme(style="darkgrid")
    spacy_counter = Counter()
    nltk_counter = Counter()
    bag_counter = Counter()
    vec_counter = Counter()
    keras_counter = Counter()
    stoken_counter = Counter()

    for _dict_ in incoming_json[1:]:
        token_list = _dict_["spacy"] + _dict_["nltk"] + _dict_["bagOwords"] + _dict_["Doc2Vec"] + _dict_["keras"] + \
                     _dict_["stoken"]
        spacy_list = _dict_["spacy"]
        nltk_list = _dict_["nltk"]
        bag_list = _dict_["bagOwords"]
        vec_list = _dict_["Doc2Vec"]
        keras_list = _dict_["keras"]
        stoken_list = _dict_["stoken"]

        for token in token_list.split(" "):
            if len(token) == 1:
                pass
            else:
                counter[token] += 1
            if re.search(r'[^./][\w+][/\w+]:', token) and len(token) > 6:
                file_path_ner[token] += 1

        for token in spacy_list.split(' '):
            if len(token) == 1:
                pass
            spacy_counter[token] += 1

        for token in nltk_list.split(' '):
            if token in punctuation:
                pass
            else:
                nltk_counter[token] += 1

        for token in bag_list.split(' '):
            if len(token) == 1:
                pass
            bag_counter[token] += 1

        for token in vec_list.split(' '):
            if len(token) == 1:
                pass
            vec_counter[token] += 1

        for token in keras_list.split(' '):
            if len(token) == 1:
                pass
            keras_counter[token] += 1

        for token in stoken_list.split(' '):
            if len(token) == 1:
                pass
            stoken_counter[token] += 1

    df = pd.DataFrame(counter.most_common(30), columns=["Token Name", "Count"])
    ner_df = pd.DataFrame(file_path_ner.most_common(30), columns=["File Path", "Count"])
    sp = pd.DataFrame(spacy_counter.most_common(30), columns=["Token Name", "Count"])
    nl = pd.DataFrame(nltk_counter.most_common(30), columns=["Token Name", "Count"])
    ba = pd.DataFrame(bag_counter.most_common(30), columns=["Token Name", "Count"])
    ve = pd.DataFrame(vec_counter.most_common(30), columns=["Token Name", "Count"])
    ke = pd.DataFrame(keras_counter.most_common(30), columns=["Token Name", "Count"])
    st = pd.DataFrame(stoken_counter.most_common(30), columns=["Token Name", "Count"])

    # counter({k: c for k, c in counter.items() if c >= 20000})

    df["Count / 100"] = df["Count"].div(100)
    ner_df["Count / 100"] = ner_df["Count"].div(100)

    fig, axes = plt.subplots(ncols=4, nrows=2, squeeze=False, figsize=(30, 20))

    sns.barplot(x="Count / 100", y="Token Name", data=df, estimator=np.median, ax=axes[0, 0]).set(
        title="Total Token Counts")
    sns.barplot(x="Count / 100", y="File Path", data=ner_df, estimator=np.median, ax=axes[1, 0]).set(
        title="File Path Counts")
    sns.barplot(x="Count", y="Token Name", data=sp, estimator=np.median, ax=axes[0, 1]).set(
        title="Spacy Token Counts")
    sns.barplot(x="Count", y="Token Name", data=nl, estimator=np.median, ax=axes[0, 2]).set(
        title="NLTK Token Counts")
    sns.barplot(x="Count", y="Token Name", data=ba, estimator=np.median, ax=axes[0, 3]).set(
        title="BagOWords Token Counts")
    sns.barplot(x="Count", y="Token Name", data=ve, estimator=np.median, ax=axes[1, 1]).set(
        title="Doc2Vec Token Counts")
    sns.barplot(x="Count", y="Token Name", data=ke, estimator=np.median, ax=axes[1, 2]).set(
        title="Keras Token Counts")
    sns.barplot(x="Count", y="Token Name", data=st, estimator=np.median, ax=axes[1, 3]).set(
        title="Stokenizer Token Counts")

    plt.savefig("_plot_all_8_token_plots.png")

    plt.subplots_adjust(wspace=4, hspace=0.75, left=10, right=10.25)
    plt.show()


# Main Function
def main():
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='json file contianing the collected tokens')
    args = parser.parse_args()
    file_name = args.file_name

    counter = Counter()
    file_path_ner = Counter()

    _ = read_json(filename=file_name)
    plot_tokens(incoming_json=_, counter=counter, file_path_ner=file_path_ner)


if __name__ == '__main__':
    main()

    # Configurations
    # python 3.plot_token_counts.py _presenting_json.json
