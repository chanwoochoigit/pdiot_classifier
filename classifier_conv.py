import numpy as np
import pandas as pd
from class_dictionary import class_dic, wrist_dic
from sklearn.model_selection import train_test_split
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

wristList=["walk_wrist_right",
            'climbing_wrist_right',
            'descending_wrist_right',
            'deskwork_wrist_right',
            'lyingLeft_wrist_right',
            'lyingRight_wrist_right',
            'lyingBack_wrist_right',
            'lyingStomach_wrist_right',
            'run_wrist_right',
            'sitForward_wrist_right',
            'sitBackward_wrist_right',
            'sitStand_wrist_right'
            ]

def filter_data(data,classlist):
    return data[data["class"].isin(classlist)]

def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot.fit_transform(integer_encoded)
    return onehot_encoded

def run_conv_model(X_train, X_test, y_train, y_test):
    num_filters = 8
    filter_size = 128
    kernel_size = 6
    pool_size = 2
    n_features = 1
    print(X_train.shape)
    print(X_test.shape)
    model = Sequential()
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

windowed_data = np.load('windowed_data_conv.npy')
class_labels = np.load('class_labels_conv.npy')
data = np.array(list(zip(windowed_data, class_labels)))
"""data[i][0]: value vectors of (36,3)"""
"""data[i][1]: class"""
# print(data)

x = []
labels = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    labels.append(data[i][1])

x = np.array(x)
y = encode_labels(labels)

print(x)
print(y)
print(x.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# run_advanced_model(X_train, X_test, y_train, y_test)
run_conv_model(X_train, X_test, y_train, y_test)
# run_target_model(X_train, X_test, y_train, y_test)

