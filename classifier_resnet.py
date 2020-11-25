# resnet model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
import matplotlib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from utils.utils import save_test_duration
matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import save_logs
from utils.utils import calculate_metrics

#https://github.com/hfawaz/dl-4-tsc/blob/master/main.py
class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=True, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # BLOCK 4

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_3)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_3)

        output_block_4 = keras.layers.add([shortcut_y, conv_z])
        output_block_4 = keras.layers.Activation('relu')(output_block_4)

        # BLOCK 5

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_4)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_4)

        output_block_5 = keras.layers.add([shortcut_y, conv_z])
        output_block_5 = keras.layers.Activation('relu')(output_block_5)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_5)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_test, y_test):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        batch_size = 32
        nb_epochs = 100

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_test, y_test, return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_test, duration)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = accuracy_score(y_test, y_pred)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

def encode_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    onehot = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot.fit_transform(integer_encoded)
    return onehot_encoded

def np_to_tensor(np_array):
  tensor = tf.convert_to_tensor(np_array, dtype=tf.float32)
  return tensor

def count_classes(data):

    print(data)
    sitstand_count = 0
    run_count = 0
    lying_count = 0
    deskwork_count = 0
    walk_count = 0
    climb_count = 0
    descend_count = 0
    active_count = 0
    inactive_count = 0
    for i in range(len(data)):
        if data[i][1] == 'sitting/standing':
            sitstand_count += 1
            inactive_count += 1
        elif data[i][1] == 'running':
            run_count += 1
            active_count += 1
        elif data[i][1] == 'lying':
            lying_count += 1
            inactive_count += 1
        elif data[i][1] == 'deskworking':
            deskwork_count += 1
            inactive_count += 1
        elif data[i][1] == 'walking':
            walk_count += 1
            active_count += 1
        elif data[i][1] == 'climbing' or data[i][1] == 'descending':
            climb_count += 1
            active_count += 1

    print('sitting/standing:{}, running:{}, lying:{}, deskwork:{}, walking:{}, climbing:{}, descending:{}'.format(
        sitstand_count, run_count, lying_count, deskwork_count, walk_count, climb_count, descend_count)
    )
    print('active:{}, inactive:{}'.format(active_count, inactive_count))

windowed_data = np.load('windowed_data_conv.npy')
class_labels = np.load('class_labels_conv.npy')

data = np.array(list(zip(windowed_data, class_labels)), dtype=object)
"""data[i][0]: value vectors of (16,3)"""
"""data[i][1]: class"""

for i in range(len(data)):
    if 'sit' in data[i][1]:
        data[i][1] = 'sitting/standing'
    elif 'run' in data[i][1]:
        data[i][1] = 'running'
    elif 'lying' in data[i][1]:
        data[i][1] = 'lying'
    elif 'deskwork' in data[i][1]:
        data[i][1] = 'deskworking'
    elif 'walk' in data[i][1]:
        data[i][1] = 'walking'
    elif 'climb' in data[i][1] or 'descend' in data[i][1]:
        data[i][1] = 'stairmoving'

print(data.shape)

x = []
labels = []
y = []
for i in range(len(data)):
    x.append(data[i][0])
    labels.append(data[i][1])

x = np.array(x)
y = encode_labels(labels)
print(x.shape)
print(y.shape)

labeling = []
for line in list(zip(labels, y)):
    if str(line) not in labeling:
        labeling.append(str(line))
print(labeling)

x_2d = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
print(x_2d.shape)
print(y.shape)

x_resampled, y_resampled = SMOTE().fit_resample(x_2d, y)

x = np.reshape(x_resampled, (x_resampled.shape[0], 16, 3))
y = y_resampled

print(x.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifier = Classifier_RESNET(output_directory='./resnet/', input_shape=(16,3), nb_classes=6)
# classifier.fit(X_train, y_train, X_test, y_test)
y_label = np.argmax(y_test, axis=1)
result = classifier.predict(X_test, y_label)
print(result)
# # predictions = np.load('resnet/y_pred.npy')
# # print(predictions)