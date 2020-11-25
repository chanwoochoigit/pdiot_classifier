import numpy as np
import pandas as pd

def string_to_num(string_list):
    num_list = []
    for str in string_list:
        num_list.append(np.double(str))
    return num_list

tdata = pd.read_csv("training_data.csv")

tdata = tdata.loc[:, ~tdata.columns.str.contains('^Unnamed')]

new_data = []

"""set data coefficients"""""""""
window_size = 64
overlap_coefficient = 16
""""""""""""""""""""""""""""""

overlap = []

for i in range(len(tdata)):
    value_vector_old = string_to_num(tdata.iloc[i]['value_vector'].replace('[','').replace(']','').replace(' ','').split(','))
    value_vector_new = []
    for j in range(len(value_vector_old)):
        if new_data == []:
            value_vector_new.append(value_vector_old[j])
            if len(value_vector_new) == window_size:
                new_data.append([value_vector_new, tdata.iloc[i]['class']])
                overlap = value_vector_new[(window_size-overlap_coefficient):]
                value_vector_new = []
        else:
            value_vector_new[:overlap_coefficient] = overlap
            value_vector_new.append(value_vector_old[j])
            if len(value_vector_new) == window_size:
                new_data.append([value_vector_new, tdata.iloc[i]['class']])
                overlap = value_vector_new[(window_size-overlap_coefficient):]
                value_vector_new = []

        print("proress...{}/{}".format(len(new_data),len(tdata)*window_size))
        print("__________________________________")

# print(new_data)
pd.DataFrame(new_data).to_csv('training_data_windowed_overlapped.csv')