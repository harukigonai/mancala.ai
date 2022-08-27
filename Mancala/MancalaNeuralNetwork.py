from tensorflow.keras.models import Model, model_from_json, clone_model
from tensorflow.keras.layers import Reshape, Activation, BatchNormalization, \
    Conv2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from os.path import exists
from threading import Lock

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


class MancalaNeuralNetwork(NeuralNetwork):
    def constructActivation(self, input):
        conv2D_layer = Conv2D(args['num_channels'], 3, padding='same')(input)
        batch_norm_layer = BatchNormalization(axis=3)(conv2D_layer)
        return Activation('relu')(batch_norm_layer)

    def constructDropout(self, num, input):
        dense_layer = Dense(num)(input)
        batch_norm_layer = BatchNormalization(axis=1)(dense_layer)
        activation_layer = Activation('relu')(batch_norm_layer)
        return Dropout(args['dropout'])(activation_layer)

    def __init__(self):
        if exists('nn_model.json'):
            self.load_model()
        else:
            self.create_nn_from_scratch()
        self.model.compile(loss=['categorical_crossentropy',
                                 'mean_squared_error'],
                           optimizer=Adam(args['lr']))

        self.predict_lock = Lock()

    def create_nn_from_scratch(self):
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

    def load_nn(self):
        nn_file = open('nn_model.json', 'r')
        nn_file_json = nn_file.read()
        nn_file.close()
        self.model = model_from_json(nn_file_json)

        self.model.load_weights("nn_model_weights.json")

    def train(self, data):
        s_li = np.array([[*sample['s'].board[0:6], sample['s'].pit_pos_1,
                          *sample['s'].board[6:12], sample['s'].pit_neg_1]
                         for sample in data]).astype('float32')
        P_li = np.array([sample['P'] for sample in data]).astype('float32')
        v_li = np.array([sample['v'] for sample in data]).astype('float32')

        self.model.fit(x=s_li, y=[P_li, v_li], batch_size=len(data), epochs=1)

    def predict(self, s):
        x = np.asarray([*s.board[0:6], s.pit_pos_1,
                        *s.board[6:12], s.pit_neg_1]).astype('float32')
        x = np.reshape(x, (1, 14)).astype('float32')

        self.predict_lock.acquire()
        pi, v = self.model.predict(x, verbose=False)
        self.predict_lock.release()

        return pi[0], v[0]

    def load_model(self):
        model_file = open('nn_model.json', 'r')
        model_json = model_file.read()
        model_file.close()
        self.model = model_from_json(model_json)

        self.model.load_weights("nn_model_weights.h5")
        print("Loaded model")

    def save_model(self):
        model_json = self.model.to_json()
        with open("nn_model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("nn_model_weights.h5")
        print("Saved model")

    def save_model_temp(self):
        model_json = self.model.to_json()
        with open("nn_model_temp.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("nn_model_temp_weights.h5")
        print("Saved model")

    def load_model_temp(self):
        model_file = open('nn_model_temp.json', 'r')
        model_json = model_file.read()
        model_file.close()
        self.model = model_from_json(model_json)

        self.model.load_weights("nn_model_temp_weights.h5")
        print("Loaded model")

    def copy_nn(self):
        nn_copy = MancalaNeuralNetwork()
        nn_copy.model = clone_model(self.model)
        nn_copy.model.set_weights(self.model.get_weights())
        return nn_copy
