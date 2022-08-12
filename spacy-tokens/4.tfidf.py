# Coding in utf-8
"""
Created on date: 19 July 2022
@authors: Timothy J. Naudet

Purpose: Perform tfidf vectorization on the tokens collcted by the tokenizers

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF, PCA
from _utilities import *

# Function to perform TFIDF Pipelining
def tfidf_clusering(X_train, y_train, X_test):
    """
    :param X_train: the X components of a labeled dataset
    :param y_train: the labels from the labeled dataset
    :param X_test: the X_test data as split from the utilities split_data_function
    :return:
    """
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf.fit_transform(X_train, y_train)
    predictions = text_clf.predict(X_test)
    return predictions


# Function to plot the clusters
def plot_top_words(model, feature_names, n_top_words, title):
    """
    :param model: the  fit model of your choosing (LDA, NMF, ...)
    :param feature_names: the fit_transformed vectorization model
    :param n_top_words: integer choice
    :param title: string for the plot title
    :return: nothing; plots the top words

    citation: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.h
    tml#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
    """

    topic_map = {
        0: "Features", 1: "Package", 2: "Version", 3: "Creation", 4: "FilePath",
        5: "Invalidation", 6: "Exceptions", 7: "Guidance", 8: "FilePath2", 9: "Corruption"
    }
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    ''' Option to label topics as the top three topics '''
    # topic_list = []
    # for topic_idx, topic in enumerate(model.components_):
    #     top_n = [feature_names[i]
    #              for i in topic.argsort()
    #              [-10:]][::-1]
    #     top_features = ' '.join(top_n)
    #     topic_list.append(f"Topic_{'_'.join(top_n[:3])}")

    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        # ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.set_title(f"Topic: {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig("_plot_NMF_TFIDF_TOP_WORDS.png")
    plt.show()

# Main function
def main():

    incoming_json = read_json("_presenting_json.json")

    # token_list = []
    # for _dict_ in incoming_json[1:]:
    #     token_list.append(_dict_["spacy"])
    #     token_list.append(_dict_['nltk'])
    #     token_list.append(_dict_['bagOwords'])
    #     token_list.append(_dict_['Doc2Vec'])
    #     token_list.append(_dict_['keras'])
    #     token_list.append(_dict_['stoken'])

    text_list = []
    for _dict_ in incoming_json[1:]:
        text_list.append(_dict_["text"])

    # token_array = pd.DataFrame(np.reshape(np.array(token_list), (len(token_list), 1)), columns=['Token_Text'])
    text_array = pd.DataFrame(np.reshape(np.array(text_list), (len(text_list), 1)), columns=['Token_Text'])

    """ TFIDF and NMF """
    print("Begining NMF...")
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # dtm is the vectorized data
    dtm = tfidf.fit_transform(text_array['Token_Text'])
    nmf_model = NMF(n_components=10, random_state=42)
    nmf_model.fit(dtm)

    topic_results = nmf_model.transform(dtm)

    plot_top_words(nmf_model, tfidf.get_feature_names_out(), 15, "NMF-TFIDF Top Words")

    """ TFDF and KMEANS """
    print("Beginning KMeans...")
    clustering_model = KMeans(
        n_clusters=10,
        max_iter=100
    )

    labels = clustering_model.fit_predict(dtm)
    print(labels.shape)

    X = dtm.todense()
    print(f"dtm shape is :{dtm.shape}")
    print(f"X.todese() shape is: {X.shape}")

    print("beginning PCA...")
    reduced_data = PCA(n_components=2).fit_transform(X)
    print(f"reduced data shape is: {reduced_data.shape}")
    fig, ax = plt.subplots()
    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
        5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
    }
    print("Beginning PCA subplot loop...")
    for index, instance in enumerate(reduced_data):
        # print instance, index, labels[index]
        pca_comp_1, pca_comp_2 = reduced_data[index]
        color = labels_color_map[labels[index]]
        ax.scatter(pca_comp_1, pca_comp_2, c=color)

    plt.savefig("_plot_TFIDF_PCA.png")
    plt.title(label="TFIDF_PCA Plot")
    # plt.xlabel("Normalized Component 1")
    # plt.ylabel("Normalized Component 2")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

    # Configurations
    # python 6.tfidf presenting_json.json
