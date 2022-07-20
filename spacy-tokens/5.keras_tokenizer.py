# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet

Purpose: Perform word2vec analysis

"""

from _utilities import keras_sequencer, git_raw_urls, request_url_data, write_json
from tqdm import tqdm

import re
import argparse


# function to perform all the tokenization and addition to the dictionary
def make_tokens(word_list):
    """
    :param word_list: the list of words selected by key selection for each dictionary in the json file
    :return: a tuple of tokens and NER from the appellate techniques
    """

    # keras section
    keras_tokens = keras_sequencer(word_list)
    for word in word_list.split(' '):
        if re.search(r'[^./][\w+][/\w+]', word) and len(word) > 6:
            keras_tokens.append(word)
        if re.search(r'[\w+][\W+][\w+]', word):
            keras_tokens.append(word)

    keras_tokens = list(set(keras_tokens))
    keras_tokens = ' '.join(keras_tokens)

    return keras_tokens


# function to produce collect the appropriate words
def words_from_json(incoming_json_text):
    """
    :param incoming_json_text: the json that we read in from the github repository; essentially a list of dictionaries,
     but in this case it has been modifed to only be the id and the text key, value pairs
    :return: a word list comprising all text in the three mentioned keys
    """
    final_transformed = {}
    for small_dict in tqdm(incoming_json_text,
                           desc=f"Progress %",
                           position=1,
                           unit=' ID',
                           mininterval=120,
                           miniters=100, leave=True, colour="magenta"):
        word_list = small_dict['text']
        keras = make_tokens(word_list)
        small_dict["keras"] = keras
        final_transformed[small_dict['id']] = small_dict

    return final_transformed


# main function
def main():
    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repository name')
    args = parser.parse_args()
    user_name = args.User_name
    repository = args.Repository
    raw_urls = git_raw_urls(user_name, repository, "error_files")
    index = 0

    # meta_urls = git_raw_urls(user_name=user_name, repository=repository, flag="docs")
    # meta_json, _ = request_url_data(meta_urls[0])
    # meta_dictionary = meta_file_splitting(meta_json)

    # for url in tqdm(raw_urls, desc="Num json files", position=0, colour="blue", unit=" Json Files"):
    for url in raw_urls:
        requested_json, requested_json_text = request_url_data(url)
        outbound_json_text = words_from_json(requested_json_text)
        for _dict_ in requested_json:
            for key in outbound_json_text.keys():
                if _dict_['id'] == key:
                    _dict_['keras'] = outbound_json_text[_dict_['id']]["keras"]

        write_json(requested_json, f"_errors_{index}_keras_tokens.json")
        index += 1


if __name__ == '__main__':
    main()

    # configurations
    # python 4.token_comparison.py TimJN17 spack-monitor-nlp
