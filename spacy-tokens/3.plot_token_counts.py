# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet...

Purpose: Plot tokens from the tokens compared

"""

import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import re
import matplotlib.pyplot as plt
from collections import Counter
from _utilities import read_json


# Function for plotting
def plot_tokens(incoming_json, counter, file_path_ner):
    sns.set_theme(style="darkgrid")
    # sns.set(rc={'figure.figsize': (15, 8.27)})

    for _dict_ in incoming_json[1:]:
        token_list = _dict_["spacy"] + _dict_["nltk"] + _dict_["bagOwords"] + _dict_["Doc2Vec"] + _dict_["keras"] + \
                     _dict_["stoken"]
        for token in token_list.split(" "):
            if len(token) == 1:
                pass
            else:
                counter[token] += 1
            if re.search(r'[^./][\w+][/\w+]:', token) and len(token) > 6:
                file_path_ner[token] += 1

    df = pd.DataFrame(counter.most_common(30), columns=["Token Name", "Count"])
    ner_df = pd.DataFrame(file_path_ner.most_common(30), columns=["File Path", "Count"])

    df["Count / 100"] = df["Count"].div(100)
    ner_df["Count / 100"] = ner_df["Count"].div(100)

    fig, axes = plt.subplots(ncols=1, nrows=2, squeeze=False)

    bp = sns.barplot(x="Count / 100", y="Token Name", data=df, estimator=np.median, ax=axes[0, 0]).set(
        title="Total Token Counts")
    ner_bp = sns.barplot(x="Count / 100", y="File Path", data=ner_df, estimator=np.median, ax=axes[1, 0]).set(
        title="File Path Counts")

    # plt.savefig("_plot_total_token_counts.png")
    plt.savefig("_plot_double_token_plot.png")

    plt.subplots_adjust(wspace=1, hspace=0.75)
    plt.axis('auto')
    plt.show()


# Main Function
def main():
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name', type=str, help='Github repository user_name')
    args = parser.parse_args()
    file_name = args.file_name

    counter = Counter()
    file_path_ner = Counter()

    # _ = []
    # for file in glob("_stoken_?_tokens.json"):
    #     _.extend(read_json(file))

    _ = read_json(filename=file_name)
    plot_tokens(incoming_json=_, counter=counter, file_path_ner=file_path_ner)


if __name__ == '__main__':
    main()

    # Configurations
    # python 3.plot_token_counts.py _presenting_json.json
