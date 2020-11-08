import numpy as np
import pandas as pd


class DataShaping(object):
    @staticmethod
    def dataAverage(lst):
        arr = np.array(lst)
        if (len(arr) == 1):
            sum_arr = arr[0]
        else:
            sum_arr = np.sum(arr, axis=0)
        return sum_arr / len(arr)

    @staticmethod
    def makeCsv(lst, cols, file_name):
        avg_arr = DataShaping.dataAverage(lst)
        csv_lst = []
        for i in range(len(avg_arr)):
            csv_lst.append([i+1, avg_arr[i]])
        df = pd.DataFrame(csv_lst, columns=cols)
        df.to_csv('output/' + file_name, index=False)
