# Coding in utf-8
"""
Created on date: 19 July 2022
@authors: Timothy J. Naudet

Purpose: Perform tfidf vectorization on the tokens collcted by the tokenizers

"""

import scipy
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from _utilities import *


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
    my_topic_dictionary = {1: 'health', 2: 'topic2', 3: 'election', 4: 'poli', 5: 'election'}
    df['Topic Label'] = df['Topic #'].map(my_topic_dictionary)
    return df


# Function to perform TFIDF Pipelining
def tfidf_clusering(X_train, y_train, X_test):
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
    text_clf.fit_transform(X_train, y_train)
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
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        # ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.set_title(f"Topic: {topic_map[topic_idx]}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# Main function
def main():
    """
    Need to read in each error-#.json file, add each set of tokens as a separate string into a corpus, and vectorize it.
    Another issue is that the data is essentially without labels; we can use topic modeling to stop this?
    :return: nothing
    """
    incoming_json = read_json("_presenting_json.json")
    token_list = []
    for _dict_ in incoming_json[1:]:
        token_list.append(_dict_["spacy"])
        token_list.append(_dict_['nltk'])
        token_list.append(_dict_['bagOwords'])
        token_list.append(_dict_['Doc2Vec'])
        token_list.append(_dict_['keras'])
        token_list.append(_dict_['stoken'])

    token_array = pd.DataFrame(np.reshape(np.array(token_list), (len(token_list), 1)), columns=['Token_Text'])

    """ TFIDF and NMF """
    print("Begining NMF...")
    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    # dtm is the vectorized data
    dtm = tfidf.fit_transform(token_array['Token_Text'])
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

    print("begining PCA...")
    reduced_data = PCA(n_components=2).fit_transform(X)
    print(f"reduced data shae is: {reduced_data.shape}")
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
    plt.show()

    ##
    # Do the dimension reduction
    # ##
    # k = 10  # number of nearest neighbors to consider
    # d = 2  # dimensionality
    # pos = manifold.Isomap(10, 2, eigen_solver='auto').fit_transform(dtm.toarray())
    #
    # ##
    # # Get meaningful "cluster" labels
    # ##
    # # Semantic labeling of cluster. Apply a label if the clusters max TF-IDF is in the 99% quantile of the whole corpus of TF-IDF scores
    # labels = tfidf.get_feature_names()  # text labels of features
    # clusterLabels = []
    # t99 = scipy.stats.mstats.mquantiles(X.data, [0.99])[0]
    # clusterLabels = []
    # for i in range(0, dtm.shape[0]):
    #     row = dtm.getrow(i)
    #     if row.max() >= t99:
    #         arrayIndex = numpy.where(row.data == row.max())[0][0]
    #         clusterLabels.append(labels[row.indices[arrayIndex]])
    #     else:
    #         clusterLabels.append('')
    # ##
    # # Plot the dimension reduced data
    # ##
    # plt.xlabel('reduced dimension-1')
    # plt.ylabel('reduced dimension-2')
    # for i in range(1, len(pos)):
    #     plt.scatter(pos[i][0], pos[i][1], c='cyan')
    #     plt.annotate(clusterLabels[i], pos[i], xytext=None, xycoords='data', textcoords='data', arrowprops=None)
    #
    # plt.show()

    # requires data with y "labels"
    # svc_tfidf = tfidf_clusering()

    # plt.scatter(x=topic_results[:, :], y=topic_results.shape[1])
    # plt.show()

    # for index, topic in enumerate(nmf_model.components_):
    #     print(f"The top 15 words for topic # {index}")
    #     print([tfidf.get_feature_names_out()[index] for index in topic.argsort()[-15:]])
    #     print("\n")
    #     print("\n")


if __name__ == '__main__':
    main()

    # Configurations
    # python 6.tfidf presenting_json.json
