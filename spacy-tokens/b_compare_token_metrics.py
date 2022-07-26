# Coding in utf-8
"""
Created on date:
@authors: Timothy J. Naudet...

Purpose:

"""

from glob import glob
import numpy as np
from _utilities import read_json, array_writing


# Function to read the files and create the array
def collect_tokens(incoming_json, final_array):
    for _dict_ in incoming_json:
        n = 0; b = 0; d = 0; k = 0; s = 0

        for word in _dict_['nltk'].split(' '):
            if word in _dict_['spacy'].split(' '):
                n += 1
        for word in _dict_['bagOwords'].split(' '):
            if word in _dict_['spacy'].split(' '):
                b += 1
        for word in _dict_['Doc2Vec'].split(' '):
            if word in _dict_['spacy'].split(' '):
                d += 1
        for word in _dict_['keras'].split(' '):
            if word in _dict_['spacy'].split(' '):
                k += 1
        for word in _dict_['stoken'].split(' '):
            if word in _dict_['spacy'].split(' '):
                s += 1

        metrics = [round(n / len(_dict_['spacy'].split(' ')) * 100, 2),
                   round(b / len(_dict_['spacy'].split(' ')) * 100, 2),
                   round(d / len(_dict_['spacy'].split(' ')) * 100, 2),
                   round(k / len(_dict_['spacy'].split(' ')) * 100, 2),
                   round(s / len(_dict_['spacy'].split(' ')) * 100, 2)]

        final_array = np.vstack((final_array,
                                 np.array([_dict_['id'], metrics[0], metrics[1], metrics[2],
                                           metrics[3], metrics[4], _dict_['text']])))

    final_array = np.delete(final_array, obj=0,  axis=0)
    return final_array


# Main Function
def main():

    final_array = np.ndarray(shape=(1, 7))
    for file in glob("_stoken_errors_?_tokens.json"):
        list_of_dicts = read_json(file)

        outbound_array = collect_tokens(list_of_dicts, final_array=final_array)
        final_array = np.vstack((final_array, outbound_array))

    final_metric = [round(float(np.mean([final_array[:, 1]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 2]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 3]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 4]], dtype=np.float64)), 2),
                    round(float(np.mean([final_array[:, 5]], dtype=np.float64)), 2)]

    # final_metric_2 = round(float(np.mean(final_array[:, 1:6], dtype=np.float64, axis=1)), 2)

    header = f"The average Percentages of overlap with the Spacy Tokens is for each column are: \n " \
             f"\t\t\t\t {final_metric[0]} \t\t  {final_metric[1]} \t\t  {final_metric[2]} \t\t  {final_metric[3]} \t\t  {final_metric[4]}\n" \
             f"The columns represent the percentage of tokens are common with the spacy tokens\n"\
             f"{'Error_ID  '.ljust(7)}   {'Spacy-NLTK'.ljust(7)}   {'Sp-BagOWs'.ljust(7)}  " \
             f"{'Sp-Doc2V '.ljust(7)}   {'Spcy-Keras'.ljust(7)}   {'Sp-Stoknzr'.ljust(7)}   " \
             f"{'Error_Text'.ljust(7)} "
    final_array = np.delete(final_array, obj=0, axis=0)
    final_array = np.vstack((np.array([['AVG VALS:', str(final_metric[0]), str(final_metric[1]),
                              str(final_metric[2]), str(final_metric[3]), str(final_metric[4]), ' ']]), final_array))
    array_writing(final_array, header=header, filename="_presenting_array.txt")


if __name__ == '__main__':
    main()

    # Configurations
    # python b_compare_token_metrics.py _stoken_errors_0_tokens.json
