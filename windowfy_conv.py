import numpy as np
import pandas as pd

def string_to_num(string_list):
    num_list = []
    for str in string_list:
        num_list.append(np.double(str))
    return num_list

tdata = pd.read_csv("training_data_conv.csv")

tdata = tdata.loc[:, ~tdata.columns.str.contains('^Unnamed')]
print(tdata)
new_data = []
class_labels = []
"""set data coefficients"""""""""
window_size = 36
overlap_coefficient = 12
""""""""""""""""""""""""""""""
value_vector_new = []
overlap = []

for i in range(len(tdata)):
    if new_data == []:
        value_vector_old = string_to_num(tdata.iloc[i]['value_vector'].replace('(','').replace(')','').replace(' ','')
                                                                      .replace('[','').replace(']','').split(','))
        print(value_vector_old)
        value_vector_new.append(value_vector_old)

        if len(value_vector_new) == 36:
            new_data.append(np.array(value_vector_new))
            value_vector_new = []
            class_labels.append(tdata.iloc[i]['class'])
    else:
        idx_overlap = i-overlap_coefficient
        value_vector_old = string_to_num(tdata.iloc[idx_overlap]['value_vector'].replace('(', '').replace(')', '').replace(' ', '')
                                                                                .replace('[', '').replace(']', '').split(','))

        print(value_vector_old)
        value_vector_new.append(value_vector_old)
        if len(value_vector_new) == 36:
            new_data.append(np.array(value_vector_new))
            value_vector_new = []
            class_labels.append(tdata.iloc[idx_overlap]['class'])
    print("proress...{}/{}".format(len(new_data),round(len(tdata)/36)))
    # print(new_data)
    print("__________________________________")

np.save('windowed_data_conv.npy', new_data)
np.save('class_labels_conv.npy', class_labels)
