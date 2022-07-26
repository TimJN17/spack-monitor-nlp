# Coding in utf-8
"""
Created on date: 8 July 2022
@authors: Timothy J. Naudet
"""

import numpy as np
import json
import yaml
import os
import re
import requests
import github
import argparse
from tensorflow.python import tf2
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import text_to_word_sequence


# Function for Argparser; nargs only use if passing in a list for the argument type
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repository name')
    args = parser.parse_args()
    return args


# Function to split the data
def split_data(X, y, test_size, random_state):
    """
    USED FOR SUPERVISED LEARNING
    :param X: the columns of the relevant data from that are 'data' and not labels; eg: X = df['review'],
    :param y: the columns of the relevant dataframe that are the labels for each row, e.g.: y = df['label']
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Function to clean dataframe
def clean_dataframe(df):
    df.dropna(inplace=True)
    blanks = []

    # df.itertuples() returns the index, the label, and the text as a three part tuple
    for i, lb, rv in df.itertuples():
        if rv.isspace():
            blanks.append(i)

    df.drop(blanks, inplace=True)
    return df


# Loading a JSON file
def load_json(file):
    f = open(file=file, mode='r')
    j = json.load(f)
    f.close()
    return j


# Class for json files
class SetEncoder(json.JSONEncoder):
    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)


# Reading a json with 'json.loads'
def read_json(filename):
    with open(filename, "r") as fd:
        content = json.loads(fd.read())
    return content


# Print to a JSON file
def write_json(content, filename):
    with open(filename, "w") as fd:
        fd.write(json.dumps(content, indent=4, cls=SetEncoder))
    fd.close()


# Printing to a text file
def print_txt_file(tuple_list_object, filename):
    # NOTE: If the output file is a JSON file, the current glob
    # process throws an error because the output file is not what we wish
    # to use for processing.
    with open(file=filename, mode='w+') as f:
        f.write(f"{'Error_ID'} {'Rank'.ljust(5)} {'Token-Lemma'.ljust(15)} {'Raw Count'}\n")
        for four_item_tup in tuple_list_object:
            f.write(
                f'{str(four_item_tup[0]).ljust(8)} {str(four_item_tup[1]).ljust(5)} {str(four_item_tup[2]).ljust(15)} {str(four_item_tup[3])}\n')
        f.close()


# Function to write an array to file
def array_writing(array, header, filename):
    """
    :param array: array of collected tokens from the files.
    :param header: the formatted string for the array text file.
    :param filename: desired name of the file.
    :return: nothing.
    """
    with open(file=filename, mode='ab') as f:
        np.savetxt(fname=filename, fmt='%10.150s', X=array, newline='\n', delimiter='   ',
                   header=header, comments='##')
    f.close()


# Function to read YAML Files
def read_yaml(filename):
    with open(filename, "r") as fd:
        content = yaml.safe_load(fd)
    return content


# Function to read text files
def read_tuple_text(file):
    """
    :param file: a file object of tuples where the error_id is the first column and the
    second column is the token itself. The count-per-id of each token is the fourth column.
    :return: an array of the files to be used for error_id manipulation in column 1.
    """
    with open(file=file, mode="r+") as f:
        data = np.genfromtxt(fname=f, skip_header=1, dtype=str)
    f.close()
    return data


# Function to return a json file from a url string
def request_url_data(url_param):
    requested_json = requests.get(url_param).json()
    final_text = []
    if type(requested_json) == list:
        for dict in requested_json:
            requested_text = {}
            requested_text['id'] = dict['id']
            requested_text['text'] = dict['text']
            final_text.append(requested_text)
    else:
        pass
    return requested_json, final_text


# Function to get a list of the contents in the repository path
def git_raw_urls(user_name, repository, flag):
    """
    :param user_name: github user_name
    :param repository: a string for the name of the repository
    :param flag: a sstring to determine which folder to enter to get which files. Choose "error_files" or "meta_file"
    :return: a list of urls that can be passed into 'requested_url_data' to get the associated json files
    """
    # instantiate a dictionary for the json files
    raw_url_list = []

    # Create a github object
    g = github.Github()

    # Access a user's github
    user = g.get_user(user_name)

    # Access  specific repository
    error_repository = user.get_repo(f"{repository}")

    # Get the content List
    if flag == "error_files":
        data = error_repository.get_contents("data")
        for item in data:
            # if "errors-" and ".json" in item.path:
            if "errors-" in item.path:
                # Here 'raw' is a list of dictionaries just like when accessing the url directly
                print(f"checking path: {item.path}")

                # Append the newly found json list of dicts ot the final dict
                raw_url_list.append(item.download_url)

    elif flag == "docs":
        data2 = error_repository.get_contents("docs")
        for item in data2:
            if "meta" in item.path:
                # Here 'raw' is a list of dictionaries just like when accessing the url directly
                print(f"checking path: {item.path}")

                # Append the newly found json list of dicts ot the final dict
                raw_url_list.append(item.download_url)

    elif flag == "spacy":
        data2 = error_repository.get_contents("spacy-tokens")
        for item in data2:
            if "errors" in item.path:
                # Here 'raw' is a list of dictionaries just like when accessing the url directly
                print(f"checking path: {item.path}")

                # Append the newly found json list of dicts ot the final dict
                raw_url_list.append(item.download_url)


    return raw_url_list


# Function to split meta.json sections into comparable token objects
def meta_file_splitting(json_dict):
    # instantiate a dictionary for the final object
    final_dict = {}
    # print(len(list(json_dict.keys())))
    for key1 in json_dict.keys():
        parsed = []
        counter_object = Counter()
        for key2 in json_dict[key1].keys():
            if "text_parsed" in key2:
                parsed.extend(json_dict[key1][key2].split(" "))
        for item in parsed:
            counter_object[item] += 1

        final_dict[json_dict[key1]["id"]] = counter_object

    return final_dict


# Function to get .py Files
def git_repository_files(user_name, repository, folder):
    """
    :param user_name: a string for the github repository user's name
    :param repository: a string for the name of the repository
    :param folder: a string for the path of the desired fodler
    :return: nothing; all desired files (.py, .txt, .md, .json) will be printed to the cwd
    """
    # Create a github object
    g = github.Github()

    # Access a user's github
    user = g.get_user(user_name)

    # Access  specific repository
    error_repository = user.get_repo(f"{repository}")

    # Get the content List
    data = error_repository.get_contents(folder)
    for item in data:
        if ".py" in item.path:
            # Here 'raw' is a list of dictionaries just like when accessing the url directly
            print(f"checking path: {item.path}")
            # write the file out
            with open(item.path + '.py', mode="w+") as f:
                f.write(item.decoded_content.decode())
            f.close()

        if ".json" in item.path:
            print(f"checking path: {item.path}")
            # write the file out
            j = requests.get(item.download_url).json()
            write_json(j, os.path.basename(item.path))

        if '.txt' in item.path:
            print(f"checking path: {item.path}")
            # write the file out
            with open(os.path.basename(item.path), mode="w+") as f:
                f.write(item.decoded_content.decode())
            f.close()

        if '.md' in item.path:
            print(f"checking path: {item.path}")
            # write the file out
            with open(os.path.basename(item.path), mode="w+") as f:
                f.write(item.decoded_content.decode())
            f.close()

    return -1

# Function to split text into batches for easier analysis
def split_into_batches(error_text, max_num_words_per_batch):
    """
    :param error_text: a sigle string that represents one document or sentence or...
    :param max_num_words_per_batch: a chosen number to limit the maximum number of calculations
    :return: a list of words plit into processable groups
    """
    words_split_on_spaces = error_text.split(' ')
    num_error_pieces = int(np.ceil(len(words_split_on_spaces) / max_num_words_per_batch))
    batches = []
    for piece_idx in range(num_error_pieces):
        start_idx = piece_idx * max_num_words_per_batch
        end_idx = (piece_idx + 1) * max_num_words_per_batch
        if end_idx > len(words_split_on_spaces):
            end_idx = len(words_split_on_spaces)
        error_piece = ' '.join(words_split_on_spaces[start_idx:end_idx])
        batches.append(error_piece)
    return batches


# function to sort the tokens
def token_sorting(id_token_object):
    # instantiate list to hold all the tuple objects
    sorted_tuples = []

    # Instantiate loop for each id
    for id_key in id_token_object.keys():

        # Instantiate lists to store objects
        tokens_unsorted = []
        counts_unsorted = []

        # Iterate through the tokens and counts for each histogram
        for token, count in id_token_object[id_key].items():
            tokens_unsorted.append(token)
            counts_unsorted.append(count)

        # convert the list objects into arrays
        counts_unsorted = np.array(counts_unsorted)
        sorted_indices = np.argsort(counts_unsorted)[::-1]

        # Create loop to enumerate through the ranks
        for rank, idx in enumerate(sorted_indices):
            sorted_tuples.append(tuple((id_key, rank + 1, tokens_unsorted[idx], counts_unsorted[idx])))
    return sorted_tuples


# function to test regrex
def regex_tester(word_list):
    # () is an explicit capture of what's inside the parenthesis
    # [] is a character class that will match anything in the ranges inserted
    for word in word_list:
        if re.search(r'[^./][\w+][/\w+]:', word):
            print(word)
    return -1

# function for keras
# def keras_tokenizer(word_list):
#     keras_tokens = Tokenizer(word_list, split=' ', lower=True)
#     keras_tokens = list(keras_tokens)
#     for word in word_list.split(' '):
#         if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
#             keras_tokens.append(word)
#     return list(set(keras_tokens))

# Function for the keras_gen from sequence
def keras_sequencer(word_list):
    keras_tokens = text_to_word_sequence(word_list, split=' ', lower=True)
    for word in word_list.split(' '):
        if re.search(r'[^./][\w+][/\w+]:', word) and len(word) > 6:
            keras_tokens.append(word)
    return list(set(keras_tokens))


if __name__ == '__main__':
    pass
