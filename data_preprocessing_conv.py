import numpy as np
import pandas as pd
# from windowfy import string_to_num

def string_to_num(string_list):
    num_list = []
    for str in string_list:
        num_list.append(np.double(str))
    return num_list

def load_and_trim_csv(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data.rename(columns={'Unnamed: 0': 'remove'}, inplace=True)
    data = data.drop('remove', axis=1)
    return data

def list_similarity(list1, list2):
    count = 0
    for element in list1:
        if element in list2:
            count += 1
    return count

class_dic = {
    'walk_chest_left':          0,
    'walk_chest_right':         1,
    'walk_pocket_left':         2,
    'walk_pocket_right':        3,
    'walk_wrist_left':          4,
    'walk_wrist_right':         5,
    'climbing_chest_left':      6,
    'climbing_chest_right':     7,
    'climbing_pocket_left':     8,
    'climbing_pocket_right':    9,
    'climbing_wrist_left':      10,
    'climbing_wrist_right':     11,
    'descending_chest_left':    12,
    'descending_chest_right':   13,
    'descending_pocket_left':   14,
    'descending_pocket_right':  15,
    'descending_wrist_left':    16,
    'descending_wrist_right':   17,
    'deskwork_chest_left':      18,
    'deskwork_chest_right':     19,
    'deskwork_pocket_left':     20,
    'deskwork_pocket_right':    21,
    'deskwork_wrist_left':      22,
    'deskwork_wrist_right':     23,
    'lyingLeft_chest_left':     24,
    'lyingLeft_chest_right':    25,
    'lyingLeft_pocket_left':    26,
    'lyingLeft_pocket_right':   27,
    'lyingLeft_wrist_left':     28,
    'lyingLeft_wrist_right':    29,
    'lyingRight_chest_left':    30,
    'lyingRight_chest_right':   31,
    'lyingRight_pocket_left':   32,
    'lyingRight_pocket_right':  33,
    'lyingRight_wrist_left':    34,
    'lyingRight_wrist_right':   35,
    'lyingBack_chest_left':     36,
    'lyingBack_chest_right':    37,
    'lyingBack_pocket_left':    38,
    'lyingBack_pocket_right':   39,
    'lyingBack_wrist_left':     40,
    'lyingBack_wrist_right':    41,
    'lyingStomach_chest_left':  42,
    'lyingStomach_chest_right': 43,
    'lyingStomach_pocket_left': 44,
    'lyingStomach_pocket_right':45,
    'lyingStomach_wrist_left':  46,
    'lyingStomach_wrist_right': 47,
    'run_chest_left':           48,
    'run_chest_right':          49,
    'run_pocket_left':          50,
    'run_pocket_right':         51,
    'run_wrist_left':           52,
    'run_wrist_right':          53,
    'sitForward_chest_left':    54,
    'sitForward_chest_right':   55,
    'sitForward_pocket_left':   56,
    'sitForward_pocket_right':  57,
    'sitForward_wrist_left':    58,
    'sitForward_wrist_right':   59,
    'sitBackward_chest_left':   60,
    'sitBackward_chest_right':  61,
    'sitBackward_pocket_left':  62,
    'sitBackward_pocket_right': 63,
    'sitBackward_wrist_left':   64,
    'sitBackward_wrist_right':  65,
    'sitStand_chest_left':      66,
    'sitStand_chest_right':     67,
    'sitStand_pocket_left':     68,
    'sitStand_pocket_right':    69,
    'sitStand_wrist_left':      70,
    'sitStand_wrist_right':     71

}

""""""""""""""""""""""""""""""""""""""""""""""""""
windowed_data = np.load('windowed_data_conv.npy')
class_labels = np.load('class_labels_conv.npy')
data = np.array(list(zip(windowed_data, class_labels)))
"""data[i][0]: value vectors of (36,3)"""
"""data[i][1]: class"""
print(data)
# windowed_data = pd.read_csv('windowed_data_conv.csv')
# windowed_data.rename(columns={'Unnamed: 0': 'remove'}, inplace=True)
# windowed_data = windowed_data.drop('remove', axis=1)
#
# windowed_data = np.array(windowed_data)
# print(windowed_data)
# training_data_processed = []
#
# for i in range(len(windowed_data)):
#     new_line = []
#     values = string_to_num(windowed_data.iloc[i][0].replace('[','').replace(']','').replace(' ','').split(','))
#     cls = windowed_data.iloc[i][1]
#     values.append(cls)
#     print("progess...{}/{}".format(i,len(windowed_data)))
#     # print('length: {}, class: {}'.format(len(values),cls))
#     # print("_____________________________________________")
#     training_data_processed.append(values)
# pd.DataFrame(training_data_processed).to_csv('training_data_overlapped_processed.csv')
#
# test = pd.read_csv('training_data_overlapped_processed.csv')
# test.rename(columns={'Unnamed: 0': 'remove'}, inplace=True)
# final_data = test.drop('remove', axis=1)
#
# final_data.to_csv('training_data_overlapped.csv')
