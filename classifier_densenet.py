import numpy as np
import pandas as pd
from class_dictionary import class_dic, wrist_dic
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow import lite
import numpy as np
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import os
import pandas
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


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
    kernel_size = 9
    pool_size = 2
    n_features = 1
    print(X_train.shape)
    print(X_test.shape)

    model = DenseNet121(input_shape=(36,3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(300, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(6, kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))


    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, shuffle=True)

    # ################plot############################################
    # print(history.history.keys())
    # #  "Accuracy"
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('training_accuracy.pdf')
    # plt.show()
    # # "Loss"
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('training_loss.pdf')
    # plt.show()
    # ##################################################################3

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    accuracy = scores[1] * 100
    print(accuracy)

    model.save('/home/chanwoo/Work/unistuff/pdiot_classifier_conv/classifier_conv.h5')

    #######convert to tflite and save
    # Convert the model.
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model.
    with open('simplified_model.tflite', 'wb') as f:
        f.write(tflite_model)
    ######################################################

windowed_data = np.load('windowed_data_conv.npy')
class_labels = np.load('class_labels_conv.npy')
data = np.array(list(zip(windowed_data, class_labels)))
"""data[i][0]: value vectors of (36,3)"""
"""data[i][1]: class"""
for i in range(len(data)):
    if 'sit' in data[i][1]:
        data[i][1] = 'sitting'
    elif 'run' in data[i][1]:
        data[i][1] = 'running'
    elif 'lying' in data[i][1]:
        data[i][1] = 'lying'
    elif 'deskwork' in data[i][1]:
        data[i][1] = 'deskworking'
    elif 'walk' in data[i][1]:
        data[i][1] = 'walking'
    elif 'climbing' in data[i][1] or 'descending' in data[i][1]:
        data[i][1] = 'stairmoves'
print(data)
sit_count = 0
run_count = 0
lying_count = 0
deskwork_count = 0
walk_count = 0
stair_count = 0
active_count = 0
inactive_count = 0
for i in range(len(data)):
    if data[i][1] == 'sitting':
        sit_count += 1
        inactive_count += 1
    elif data[i][1] == 'running':
        run_count += 1
        active_count +=1
    elif data[i][1] == 'lying':
        lying_count += 1
        inactive_count += 1
    elif data[i][1] == 'deskworking':
        deskwork_count += 1
        inactive_count += 1
    elif data[i][1] == 'walking':
        walk_count += 1
        active_count += 1
    elif data[i][1] == 'stairmoves':
        stair_count += 1
        active_count += 1

print('sitting:{}, running:{}, lying:{}, deskwork:{}, walking:{}, stairmoves:{}'.format(sit_count, run_count, lying_count, deskwork_count, walk_count, stair_count))
print('active:{}, inactive:{}'.format(active_count, inactive_count))
print(data.shape)

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
labeling = []
for line in list(zip(labels, y)):
    if str(line) not in labeling:
        labeling.append(str(line))
print(labeling)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# run_advanced_model(X_train, X_test, y_train, y_test)
run_conv_model(X_train, X_test, y_train, y_test)
# run_target_model(X_train, X_test, y_train, y_test)

