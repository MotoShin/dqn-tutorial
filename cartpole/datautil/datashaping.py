import numpy as np
import pandas as pd


class DataShaping(object):

    @staticmethod
    def makeCsv(lst, kind, file_name):
        csv_lst = []
        cols = ['episode']
        for epi in range(len(lst[0])):
            cols.append("{}sim_{}".format(epi+1, kind))

        for i in range(len(lst)):
            one_line = [i+1]
            for k in range(len(lst[i])):
                one_line.append(lst[i][k])
            csv_lst.append(one_line)
        df = pd.DataFrame(csv_lst, columns=cols)
        df.to_csv('output/' + file_name, index=False)
