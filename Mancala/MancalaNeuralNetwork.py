from xml.dom import ValidationErr
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Reshape, Activation, BatchNormalization, \
    Conv2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from os.path import exists

from NeuralNetwork import NeuralNetwork

import numpy as np

args = {
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
}

MANCALA_MODEL = "nn_model.json"
MANCALA_WEIGHTS_PERM = "nn_weights_perm.h5"
MANCALA_WEIGHTS_CURR = "nn_weights_curr.h5"
MANCALA_WEIGHTS_NEW = "nn_weights_new.h5"

WEIGHTS_NAME_TO_FILE = {
    "perm": MANCALA_WEIGHTS_PERM,
    "curr": MANCALA_WEIGHTS_CURR,
    "new": MANCALA_WEIGHTS_NEW
}


class MancalaNeuralNetwork(NeuralNetwork):

    def __init__(self, load_weights=None, train_data=None):
        self.create_model()

        # load weights
        if load_weights is not None and train_data is not None:
            err = "Only one parameter may be not None"
            raise ValidationErr(err)
        if load_weights is not None and load_weights in WEIGHTS_NAME_TO_FILE:
            file_name = WEIGHTS_NAME_TO_FILE[load_weights]
            if exists(file_name):
                self.model.load_weights(file_name)
        elif train_data is not None:
            self.train(train_data)
        else:
            err = "load_weights must be perm, curr, or new"
            raise TypeError(err)

    def constructActivation(self, input):
        conv2D_layer = Conv2D(args['num_channels'], 3, padding='same')(input)
        batch_norm_layer = BatchNormalization(axis=3)(conv2D_layer)
        return Activation('relu')(batch_norm_layer)

    def constructDropout(self, num, input):
        dense_layer = Dense(num)(input)
        batch_norm_layer = BatchNormalization(axis=1)(dense_layer)
        activation_layer = Activation('relu')(batch_norm_layer)
        return Dropout(args['dropout'])(activation_layer)

    def create_model(self):
        # Neural Net
        self.input_boards = Input(shape=(14, ))

        x_image = Reshape((2, 7, 1))(self.input_boards)
        h_conv1 = self.constructActivation(x_image)
        h_conv2 = self.constructActivation(h_conv1)
        h_conv3 = self.constructActivation(h_conv2)
        h_conv4_flat = Flatten()(h_conv3)
        s_fc1 = self.constructDropout(1024, h_conv4_flat)
        s_fc2 = self.constructDropout(512, s_fc1)
        self.pi = Dense(12, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'],
                           optimizer=Adam(args['lr']))

    def train(self, data):
        s_li = np.array([[*sample['s'].board[0:6], sample['s'].pit_pos_1,
                          *sample['s'].board[6:12], sample['s'].pit_neg_1]
                         for sample in data]).astype('float32')
        P_li = np.array([sample['P'] for sample in data]).astype('float32')
        v_li = np.array([sample['v'] for sample in data]).astype('float32')

        self.model.fit(x=s_li,
                       y=[P_li, v_li],
                       batch_size=len(data),
                       epochs=1,
                       workers=4,
                       use_multiprocessing=False)

    def predict(self, s):
        x = np.asarray([*s.board[0:6], s.pit_pos_1,
                        *s.board[6:12], s.pit_neg_1]).astype('float32')
        x = np.reshape(x, (1, 14)).astype('float32')

        pi, v = self.model.predict(x, verbose=False)
        return pi[0], v[0]

    def save_model(self):
        model_json = self.model.to_json()
        with open(MANCALA_MODEL, "w") as json_file:
            json_file.write(model_json)

    def save_weights(self, with_specified_weights_if_exist):
        if with_specified_weights_if_exist in WEIGHTS_NAME_TO_FILE:
            file_name = WEIGHTS_NAME_TO_FILE[with_specified_weights_if_exist]
            self.model.save_weights(file_name)
        else:
            err = "with_specified_weights_if_exist must be perm, curr, or new"
            raise TypeError(err)

    def copy_nn(self):
        nn_copy = MancalaNeuralNetwork()
        nn_copy.model = clone_model(self.model)
        nn_copy.model.set_weights(self.model.get_weights())
        return nn_copy
