# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet

Purpose: perform named entity recognition.

"""

import re
import spacy
import nltk
import time
import stanza
from _utilities import split_into_batches, git_raw_urls, request_url_data, print_txt_file, token_sorting, parse_args
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from collections import Counter
from spacy.pipeline.ner import make_ner
# from ._parser_internals.transition_system import TransitionSystem
from spacy.pipeline import EntityRecognizer


# function for spacy NER
def spacy_NER(incoming_json):
    # https://spacy.io/usage/linguistic-features

    # Create the configuration for the model
    config = {
        "moves": None,
        "update_with_oracle_cut_size": 100,
        "model": DEFAULT_NER_MODEL,
        "incorrect_spans_key": "incorrect_spans",
    }

    # Create the nlp object
    nlp = spacy.load("en_core_web_sm")
    # nlp = en_core_web_sm.load()
    # ner = make_ner(nlp=nlp, model=DEFAULT_NER_MODEL)
    # ner = nlp.add_pipe("ner", config=config)

    temporary_dictionaries = {}
    for _dict_ in incoming_json:
        word_list = _dict_['text']
        word_list += _dict_['pre_context']
        word_list += _dict_['post_context']
        temporary_dictionaries[_dict_['id']] = word_list

    final_ents = {}

    for key in temporary_dictionaries.keys():
        counter = Counter()
        _words_ = nlp(temporary_dictionaries[key])
        for ent in _words_.ents:
            if re.match('[a-zA-Z]+$', ent.text):
                counter[str(ent.label_) + str(ent.text)] += 1

        final_ents[key] = counter

    return final_ents


# function for NLTK NER
def nltk_NER(incoming_json):
    # https://github.com/ytang07/intro_nlp/blob/main/nltk/nltk_ner.py
    # https://nanonets.com/blog/named-entity-recognition-with-nltk-and-spacy/#how-to-build-or-train-ner-model

    temporary_dictionaries = {}
    final_dict = {}
    for _dict_ in incoming_json:
        word_list = _dict_['text']
        word_list += _dict_['pre_context']
        word_list += _dict_['post_context']
        temporary_dictionaries[_dict_['id']] = word_list
    for key in temporary_dictionaries.keys():
        nltk_counter = Counter()
        tokens = nltk.word_tokenize((temporary_dictionaries[key]))
        pos_tagged = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(pos_tagged)
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                nltk_counter[chunk.label() + ' ' + chunk[0][0] + ' ' + chunk[0][1]] += 1

        final_dict[key] = nltk_counter

    return final_dict


# function for stanza
def stanza_NER(incoming_json):
    # https://github.com/ytang07/intro_nlp/blob/main/coreNLP/ner.py
    nlp = stanza.Pipeline(lang='en', processors="tokenize,ner")
    doc = nlp('some text')
    print(*[f"entity: {ent.text}\ttype: {ent.type}" for sent in doc.sentences for ent in sent.ents], sep='\n')
    return -1


def main():
    starttime = time.time()
    args = parse_args()
    user_name = args.User_name
    repository = args.Repository

    raw_urls = git_raw_urls(user_name, repository, "error_files")
    final_list = []
    final_tuple_list = []
    index = 1

    for url in raw_urls:
        requested_json = request_url_data(url)
        # entity_dict = spacy_NER(requested_json)
        nltk_dict = nltk_NER(requested_json)
        # final_list.append(entity_dict)
        final_list.append(nltk_dict)

    for _dict_ in final_list:
        final_tuple_list.extend(token_sorting(_dict_))
        print(f"successfully extended {index} urls.")
        print(f"Duration for {index} urls {(time.time() - starttime) / 60} minutes ")
        index += 1
    print_txt_file(final_tuple_list, "_3.nltkNER_tokens.txt")


if __name__ == '__main__':
    main()

    # Configurations
    # python 3.namedEntReg.py TimJN17 spack-monitor-nlp
