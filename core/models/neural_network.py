import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from time import time
import numpy as np
import pickle
import os

class FeatureMaskLayer(keras.layers.Layer):
    def __init__(self, n_features):
        super(FeatureMaskLayer, self).__init__()
        self.mask = np.ones(shape=(n_features))

    def call(self, inputs):
        return inputs * self.mask

    def update_mask(self, mask):
        if not type(mask) is np.array:
            self.mask = np.array(mask)

        else:
            self.mask = mask

    def get_config(self):
        data = {'mask': self.mask}

        return data

class NeuralNetwork():
    """Add explanation.
    """

    possible_data_types = ['counts', 'normlog']
    data_type = 'normlog'
    input_as_adata = False

    def save(self, save_path, name):
        timestamp = str(time()).replace('.', '_')
        model_path = os.path.join(
            save_path, 
            'models',
            name,
            timestamp
        )
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        settings_dict = {'classifier_type': 'NN'}
        for key in self.__dict__.keys():
            if key in ['model']:
                self.__dict__[key].save(os.path.join(model_path, 'model.SavedModel'))
                settings_dict[key] = os.path.join(model_path, 'model.SavedModel')

            elif key in ['activation_function', 'optimizer']:
                pass

            else:
                settings_dict[key] = self.__dict__[key]

        with open(os.path.join(model_path, 'classifier_settings.pickle'), 'wb') as f:
            pickle.dump(settings_dict, f)

    def __init__(
        self, 
        n_input, 
        n_output, 
        hidden_layers=[64, 64],
        learning_rate=0.01,
        momentum=0.9,
        loss_function='categorical_crossentropy',
        dropout=0.4,
        max_epochs=1000,
        batch_size=40,
        **kwargs
    ):
        """Add explanation.
        """

        self.n_input = n_input
        self.n_output = n_output
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.loss_function = loss_function
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.activation_function = lambda x: keras.activations.relu(x, alpha=0.1)
        self.histories = []
        self.init_model()

    def init_model(self):
        self.model = keras.models.Sequential()
        layers = [self.n_input] + self.hidden_layers + [self.n_output]
        layers_in_out = list(zip(layers[:-1], layers[1:]))
        for idx, layer in enumerate(layers_in_out):
            if idx == len(layers_in_out) - 1: # final layer
                activation = 'softmax'
                dropout = 0.0

            else:
                activation = self.activation_function
                dropout = self.dropout

            if idx == 0:
                self.model.add(keras.Input(
                    shape=(layer[0], )))
                # Lambda layer with mask allows for turning input nodes on/off in the future
                # Enables flexible, iterative feature selection
                self.model.add(FeatureMaskLayer(layer[0]))

            self.model.add(keras.layers.Dense(
                input_shape=(layer[0], ),
                units=layer[1],
                kernel_initializer='glorot_uniform',                                          
                bias_initializer='zeros',
                activation=activation))
            if dropout > 0.0:
                self.model.add(keras.layers.Dropout(dropout))

        self.optimizer = keras.optimizers.SGD(
            learning_rate = self.learning_rate, 
            momentum = self.momentum)
        self.model.compile(
            optimizer = self.optimizer, 
            loss = self.loss_function,
            metrics=['accuracy'])

    def train(self, x, y_onehot, y_int, plot=False, **kwargs):
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=10,
            restore_best_weights=True,
            verbose=1)
        reduce_LR_plateau_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            factor=0.2,
            verbose=1)
        x_train, x_val, y_train, y_val = train_test_split(
            x, 
            y_onehot, 
            stratify=y_int, 
            test_size=0.2, 
            random_state=42)
        print('Before fit')
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs = self.max_epochs, 
            verbose = 0,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping_callback, reduce_LR_plateau_callback])
        print('After fit')
        self.histories.append(history.history)
        if plot:
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

    def predict(self, input_vec):
        """Calculate and return label prediction of trained model for an input
        vector input_vec (dtype=int)"""

        # Returns only absolute decisions, nothing known about the difference 
        # in prediction confidence 
        pred_vec = np.argmax(self.model.predict(input_vec), axis = -1)

        return pred_vec

    def predict_proba(self, input_vec):
        """Calculate and return probabilities of label predictions of trained model for an input
        vector input_vec (dtype=int)"""

        proba_pred_vec = self.model.predict(input_vec)

        return proba_pred_vec

    def validate(self, x, y_int, **kwargs):
        """Add explanation.
        """

        y_preds = self.predict(x)
        def calc_acc(pred_vec, known_vec):
            if type(pred_vec) == type(known_vec):
                acc = np.sum(pred_vec == known_vec, axis = 0) / len(known_vec)

            else:
                raise ValueError('self.validate: Error! \
                    Comparison of different label encoding!')

            return acc

        acc = calc_acc(y_preds, y_int)
        con_mat = confusion_matrix(
            y_true=y_int, 
            y_pred=y_preds, 
            normalize = 'true')

        return acc, con_mat

    def update_feature_mask(self, mask):
        for layer in self.model.layers:
            if type(layer) is FeatureMaskLayer:
                layer.update_mask(mask)

    def reset_output(self, n_output):
        self.n_output = n_output
        input_shape = (self.model.layers[-1].input.shape[1], )
        activation = self.model.layers[-1].activation
        self.model.pop()
        self.model.add(keras.layers.Dense(
                        input_shape=input_shape,
                        units=n_output,
                        kernel_initializer='glorot_uniform',                                          
                        bias_initializer='zeros',
                        activation=activation))