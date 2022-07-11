# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet

Purpose: Perform bagOfWords.feature_extraction for tokenization on the errors-*.json file

"""

import argparse
import time
from _utilities import split_into_batches, git_raw_urls, request_url_data, print_txt_file, token_sorting
from river import feature_extraction


# Function to count tokens in sentences
def bagOwords(dict_of_words):
    """
    :param dict_of_words: a dicinotary of words;
    :return: a dictionary of the transformed values;keys are the error ids and the values are tuples of the counts and words
    documentation: https://riverml.xyz/dev/api/feature-extraction/BagOfWords/
    NOTE: The transform_one method applies to only one document in a corps whereas the transform_many applies to all
    documents in the corpus.
    """
    # to be used with learn_one and transform_one
    # tokenizer can be a regrex expression
    model = feature_extraction.BagOfWords(lowercase=True, strip_accents=True)
    final_transformed = {}
    for small_dict in dict_of_words:
        word_list = small_dict['text']
        word_list += small_dict['pre_context']
        word_list += small_dict['post_context']
        temporary_dict = model.transform_one(word_list)
        final_transformed[small_dict['id']] = temporary_dict

    return final_transformed


def main():
    starttime = time.time()

    # Argparser, nargs only use if passing in a list for the argument type
    parser = argparse.ArgumentParser()
    parser.add_argument('User_name', type=str, help='Github repository user_name')
    parser.add_argument('Repository', type=str, help='Repository name')
    args = parser.parse_args()
    user_name = args.User_name
    repository = args.Repository

    raw_urls = git_raw_urls(user_name, repository, "error_files")
    final_list = []
    final_tuple_list = []
    index = 1

    for url in raw_urls:
        requested_json = request_url_data(url)
        final_dictionary = bagOwords(requested_json)
        final_list.append(final_dictionary)

    for _dict_ in final_list:
        final_tuple_list.extend(token_sorting(_dict_))
        print(f"successfully extended {index} urls.")
        print(f"Duration for {index} urls {(time.time() - starttime) / 60} minutes ")
        index += 1
    print_txt_file(final_tuple_list, "_2.bagOwords_tokens.txt")


if __name__ == '__main__':
    main()

    # configurations
    # python 2.bagOwords.py TimJN17 spack-monitor-nlp
