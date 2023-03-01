import pandas as pd
import sys
import logging
import numpy as np
import math
import os


def test_args(X):

    if len(X) != 4:
        print("""
        Script to automatically parse the results of voting student council elections as of 2019/02/13.
        USAGE:
        python voting_results.txt <csv_file> <log_file> <out_path>""")
        sys.exit()

def set_logger(logfile):
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def read_data(infile, cols):
    """
    given the input voting file, returns the data frame with extra columns cols
    """
    df = pd.read_csv(infile, header=None, index_col=False).T
    df.columns = cols + list(range(1, len(df.columns) - 1))

    return df

def data_splitter(df, col_idx):
    """
    given the data frame of votes, return a list of data frames per position, splitted on the column col_idx
    """

    labels = df.loc[:, col_idx]
    outlist = []

    for lab in set(labels):
        between = labels == lab
        outlist.append(df[between])

    return outlist

def count_votes(df_list, out_folder, voting_function):
    """
    from the list of dataframe of individual positions, count the votes for each one using voting_function and returns a report
    """
    report = ""
    logging.info("starting the voting procedure using the voting function: {}".format(voting_function.__name__))
    for df in df_list:
        # current position
        current_pos = str(df.iloc[0,0])
        logging.info("processing votes for the position: {}".format(current_pos))

        pos_results, new_df = voting_function(df, current_pos)
        report += pos_results
        new_df.to_csv("{}/{}.csv".format(out_folder, current_pos), sep=",")


    return report

def get_index(d,v):
    """
    returns the key for value v
    :param d:
    :param v:
    :return:
    """
    if list(d.values()).count(v) == 1:
        return [key for key, val in d.items() if val == v][0] # index of v
    else:
        return -1

def modify_col(c, b):
    """
    gets column content and banned set, modify the col so to have a new number 1
    :param c:
    :param b:
    :return:
    """

    # shrink the vote if there is no 1
    while list(c.values()).count("1") != 1 or get_index(c, "1") in b:
        vals = c.values()
        if len([x for x in vals if x == float('nan') or x != x or x == ""]) == len(vals) or list(vals).count('1') > 1:  # empty col or more than 1 pref
                return ""

        # if there is no 1
        elif "1" not in vals:
            for k in c:
                if not c[k] == "":
                    if not math.isnan(float(c[k])):
                        c[k] = str(int(c[k])-1)

        # if 1 is a banned guy, remove and shrink
        idx = get_index(c, "1")
        if idx in b:
            idx = get_index(c, "1")
            c[idx] = ""

    return c


def voting_function(df, position):
    """
    first voting function, given a position specific data frame, returns the results and modified data frame.
    :return:
    """
    logger = logging.getLogger("simple_example")

    # candidates
    candidates = dict(zip(df.index, df["candidate"]))
    logger.info("The candidates are: {}".format(",".join(candidates.values())))

    # count result init
    count_result = {c: 0 for c in candidates.values()}

    # find winner loop
    store_cols = {c: [] for c in candidates.values()}
    logger.info("counting votes")
    winner = False
    df_dict = df.to_dict()
    banned = set()  # contains the excluded because least
    while not winner:
        for col in df.iloc[:, 2:]:
            if col in df_dict:
                vote_col = df_dict[col]
                # remove the column if there is more than 1 or no 1
                if list(vote_col.values()).count("1") != 1:
                    del(df_dict[col])
                    continue
                else:
                    idx = get_index(vote_col, "1")
                    count_result[candidates[idx]] += 1
                    store_cols[candidates[idx]].append(col)
        # check for a winner
        tot_sum = sum(count_result.values())
        for k, v in count_result.items():
            if v > tot_sum - v:
                winner = k
                rep = "found a winner for position '{}': {}, with {} out of {} votes\n".format(position, winner, v, tot_sum)
                logger.info(rep)
                return rep, pd.DataFrame.from_dict(df_dict)
        # no winner found
        logger.info("winner not found for position {}".format(position))
        least = min({k:v for k,v in count_result.items() if get_index(candidates, k) not in banned}, key=count_result.get)
        banned.add(get_index(candidates, least))
        logger.info("least number of votes found for {} with {} out of {}". format(least, count_result[least], tot_sum))
        count_result = {c: 0 for c in candidates.values()}
        for sub_col in store_cols[least]:
            if sub_col in df_dict:
                new_col = modify_col(df_dict[sub_col], banned)
                if not new_col:
                    del(df_dict[sub_col])
                else:
                    df_dict[sub_col] = new_col
        store_cols = {c: [] for c in candidates.values()}


def main(arg):

    # test args
    test_args(arg)

    # outfolder
    output_folder = arg[3]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # logging stuff
    logger = set_logger(arg[2])

    logger.info("ARGS seems fine. Logging working, let's start the work...")

    # read data
    voting_data = read_data(arg[1], ["position", "candidate"])

    # create list of DF per position
    pos_df_list = data_splitter(voting_data, "position")

    # apply voting function to each element of the list
    report = count_votes(pos_df_list, output_folder, voting_function)

    print(report)


if __name__ == '__main__':
    main(sys.argv)