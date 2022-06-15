
"""
Purpose: Compare different tuple outputs. 
    This script will initially only compare the different identified tokens of the two different arrays.
    This file will report similar tokens and dissimilar tokens.
    Output file is the following:
    Headers
    hared tokens, spacy only tokens, word2vec only tokens, percentage shared
"""

import sys
import numpy as np
import os
import time


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


# Function to write an array to file
def array_writing(array, header, filename):
    """
    :param array: array of collected tokens from the files.
    :param header: the formatted string for the array text file.
    :param filename: desired name of the file.
    :return: nothing.
    """
    with open(file=filename, mode='ab') as f:
        np.savetxt(fname=filename, fmt='%15.20s', X=array, newline='\n', delimiter=' ',
                   header=header, comments='##')
    f.close()


# Function to write percentages
def write_percentages(dictionary):
    """
    :param dictionary: a dictionary of the error_id and the percentage of overlapping tokens for
    each error.
    :return: nothing.
    """
    with open("percentage_file.txt", mode="a+") as f:
        # Write the average
        f.write(f"The average percentage of similar words was: "
                f"{round(sum(dictionary.values())/len(dictionary.keys()), 2)}% \n\n")
        # Write the median
        f.write(f"The median percentage of similar words was: "
                f"{round(float(np.mean(np.array(list(dictionary.values())))), 2)}% \n\n")
        # Write the percentage for each file
        for key in dictionary.keys():
            f.write(f"Error id {key} has overlapping token percentage of: {dictionary[key]:.2f}%\n")
    f.close()


# Function to separate the lines by error id and compare them
def error_comparison(array_1, array_2):
    """
    :param array_1: the spacy token array created using the above "read_tuple_text" function.
    :param array_2: the word2vec token array created using the above "read_tuple_text" function.
    :return: (1) a final array to be printed using "np.savetxt" to avoid file overwriting.
    common in "np.savetxt" iterative saving. Using "np.savetxt" and not
    a for loop because it's a very large array. (2) a dictionary of error_id as the key and the
    percentage of overlapping tokens as the value for each error_id.

    NOTE: Because of sorting as a "set", the numerical sequence is not intuitive. 
    """

    # Final Array & dictionary instantiation
    final_array = np.zeros((1, 3))
    string_dict = {}

    # Loop for each number in the set of error_ids; Note that many error_id's are missing.
    for num in sorted(set(array_1[:, 0])):

        # Convert string to int
        num = int(num)

        # Select from each array the errors with teh same id
        next_1 = array_1[array_1[:, 0] == str(num), :]
        next_2 = array_2[array_2[:, 0] == str(num), :]

        # Combine the arrays to determine uniqueness
        combined_array = np.hstack((next_1[:, 2], next_2[:, 2]))

        # Create a set of all non-unique tokens form both tokenization methods
        combined_list = list(combined_array)
        combined_set = set(combined_list)

        # Calculate the percentage of tokens that are the same
        percentage_same = (len(combined_set)/len(combined_list))*100

        # Since the plan is to use "np.vstack" and "np.hstack", need arrays of equal dimensions
        # Calculate the differences between the combined array and the spacy array
        diff = np.array([list(combined_set)]).T.shape[0] - next_1[:, 2].shape[0]
        # Create an array to pad token arrays using empty spaces
        diff_array = np.full(shape=(abs(diff), 4), dtype=str, fill_value=' ')

        # Check to see if the difference is positive or negative
        if diff < 0:
            # If a negative differences, the padding in the difference array must
            # be added to the set of non-unique tokens
            diff_array = np.reshape(diff_array[:, 2], (diff_array.shape[0], 1))
            set_array = np.vstack((np.array([list(combined_set)]).T, diff_array))
            next_1 = np.reshape(next_1[:, 2], (next_1[:, 2].shape[0], 1))
            total_array = np.hstack((set_array, next_1))
        else:
            # A positive difference indicates that the spacy array requires padding
            next_1 = np.vstack((next_1, diff_array))
            next_1 = np.reshape(next_1[:, 2], (next_1[:, 2].shape[0], 1))
            total_array = np.hstack((np.array([list(combined_set)]).T, next_1))

        # Calculate separate Variables for the same purpose for the word2vec method
        # NOTE: switch to the "total_array" because no loger working with the
        # horizontally, originally compiled "set" array
        diff_2 = total_array.shape[0] - next_2[:, 2].shape[0]
        diff_2_array = np.full(shape=(abs(diff_2), 4), dtype=str, fill_value=' ')

        # Check to see if the difference is positive or negative
        if diff_2 < 0:
            # If a negative differences, the padding in the difference array must
            # be added to the set of non-unique tokens
            diff_2_array = np.reshape(diff_2_array[:, 0:2], (diff_2_array.shape[0], 2))
            set_array = np.vstack((total_array, diff_2_array))
            next_2 = np.reshape(next_2[:, 2], (next_2.shape[0], 1))
            total_array = np.hstack((set_array, next_2))
        else:
            # A positive difference indicates that the spacy array requires padding
            next_2 = np.vstack((next_2, diff_2_array))
            next_2 = np.reshape(next_2[:, 2], (next_2.shape[0], 1))
            total_array = np.hstack((total_array, next_2))

        # Header part of the array to separate each error_id
        header_array = np.array([f"--ERROR_ID {num}", "Percentage:", str(round(percentage_same, 2))+'%'])
        total_array = np.vstack((header_array, total_array))

        # Final array section
        final_array = np.vstack((final_array, total_array))

        # Percentage storage
        string_dict[num] = percentage_same

        # Terminal Print Statements for progress
        if num % 1000 == 0:
            print(f"Iteration number: {num}")
            print(f"Elapsed time is: {round((time.time() - start_time)/60, 2)} minutes.")

    # Returnable object
    final_array = np.delete(final_array, obj=0, axis=0)

    return final_array, string_dict


# Main section
if __name__ == "__main__":

    # Start_time
    start_time = time.time()

    # System arguments
    file_name = sys.argv[1]
    file_1 = sys.argv[2]
    file_2 = sys.argv[3]

    # Main Function Calls
    file_1_array = read_tuple_text(file_1)
    file_2_array = read_tuple_text(file_2)
    last_array, dict_ = error_comparison(file_1_array, file_2_array)
    end_time = time.time()

    # Write Comparison File
    if os.path.isfile(file_name):
        os.remove(file_name)
    header = f" File for Token Characterization \n Columns are: \n {'Common_Tokens'.ljust(10)} " \
                f"{'Unique_Spacy_Tokens'.ljust(5)} {'Unique_Word2Vec_tokens'.ljust(5)} \n"
    array_writing(last_array, header, file_name)

    # Write Percentage Files
    if os.path.isfile("percentage_file.txt"):
        os.remove("percentage_file.txt")
    write_percentages(dict_)

    # Total Time
    print(f"The total time was: {(end_time - start_time) / 60} minutes")

    # Configurations
    # python llnlproject_file_comparison.py token_comparison.txt json_file_spacy_comparison.txt json_file_meta_comparison.text
    # json_file_spacy_comparison.txt
    # json_file_meta_comparison.text
