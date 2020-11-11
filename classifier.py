import numpy as np
import pandas as pd
from class_dictionary import class_dic, wrist_dic
from sklearn.model_selection import train_test_split
import pandas
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Conv1D, MaxPooling1D, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
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

#baseline model for performance comparison
def run_baseline_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(60, input_dim=36, activation='relu'))
    model.add(Dense(60, input_dim=36, activation='relu'))
    model.add(Dense(60, input_dim=36, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def run_advanced_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(512, input_dim=64, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(350, input_dim=24, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(350, input_dim=24, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(350, input_dim=24, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(350, input_dim=24, kernel_initializer='he_uniform', activation='relu'))

    model.add(Dense(12, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
    # Plot history: MAE
    plt.plot(history.history['accuracy'], label='MAE (training data)')
    plt.title('Training')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def run_conv_model(X_train, X_test, y_train, y_test):
    num_filters = 8
    filter_size = 64
    kernel_size = 3
    pool_size = 2
    n_features = 1
    """reshape input"""
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
    print(X_train.shape)
    print(X_test.shape)
    model = Sequential()
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# print(X_train)
# print(y_train)

# baseline = baseline_model()
# baseline.fit(X_train, y_train, epochs=100, batch_size=4, validation_split=0.2)
# estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X_test, y_test, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# advanced = advanced_model()
# advanced.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
# estimator = KerasClassifier(build_fn=advanced, epochs=100, batch_size=32, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X_test, y_test, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


"""remove unnamed column and rename the last column's name to class"""
data = pd.read_csv('training_data_overlapped.csv')
data.rename(columns={'Unnamed: 0': 'remove'}, inplace=True)
data.rename(columns={'64': 'class'}, inplace=True)
data = data.drop('remove', axis=1)
print(data)

"""prepare wrist_data"""""""""""""""""
wrist_data = pd.DataFrame(filter_data(data, wristList))
print(wrist_data)
x_wrist = np.array(wrist_data.iloc[:, :-1].values)
print(x_wrist)
print(x_wrist.shape)
window_size = 36

y_wrist_single = []
for cls in wrist_data['class']:
    y_wrist_single.append(wrist_dic.get(cls))
print(pd.DataFrame(y_wrist_single))
"""Create y filled with zeros"""
y_wrist = np.zeros((x_wrist.shape[0], 12))
"""fill in real values"""
for i in range(len(y_wrist)):
    # print("progress...{}/{}".format(i,len(y_wrist)))
    idx = y_wrist_single[i]
    y_wrist[i][idx] += 1

y_wrist = pd.DataFrame(y_wrist)

print(x_wrist)
print(x_wrist.shape)
print(y_wrist)
print(y_wrist.shape)
print("________________________________________________________")
"""prepare wrist_data"""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(x_wrist, y_wrist, test_size=0.2)

# run_advanced_model(X_train, X_test, y_train, y_test)
run_conv_model(X_train, X_test, y_train, y_test)
# run_target_model(X_train, X_test, y_train, y_test)

