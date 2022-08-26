import tensorflow.keras as keras
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split

class DenseBase():
    """
    """

    possible_data_types = ['counts', 'normlog']

    def train(
        self,
        x,
        y_onehot,
        y_int,
        batch_size=40,
        epochs=1000,
        verbose=0,
        plot=False,
        **kwargs):

        if hasattr(self, 'callbacks'):
            callbacks = self.callbacks

        else:
            callbacks = []

        if not hasattr(self, 'histories'):
            self.histories = []

        x_train, x_val, y_train, y_val = train_test_split(
            x, 
            y_onehot, 
            stratify=y_int, 
            test_size=0.2, 
            random_state=42)
        # TODO
        # Make sure this is transferable to torch
        history = self.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs, 
            verbose=verbose,
            validation_data=(x_val, y_val),
            callbacks=callbacks)
        self.histories.append(history)
        if plot:
            self.plot_training(history, 'accuracy')
            self.plot_training(history, 'loss')

    def predict(self, x):
        return self.model.predict(x)

    def plot_training(self, history, to_plot):
        plt.plot(history.history[f'{to_plot}'])
        plt.plot(history.history[f'val_{to_plot}'])
        plt.title(f'model {to_plot}')
        plt.ylabel(f'{to_plot}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

class DenseKeras(keras.Model, DenseBase):
    """
    TODO:
    Deal with history
    Deal with plotting and callbacks, train_val split now outside of network class
    """

    def __init__(
        self, 
        n_input=None, 
        n_output=None, 
        model=None,
        early_stopping=True,
        reduce_LR_plateau=True,
        sequential_kwargs={}, 
        **kwargs):

        super().__init__()
        self.possible_data_types = tf.Variable(['counts', 'normlog'])
        self.data_type = tf.Variable('normlog')
        if model is None:
            if n_input is None or n_output is None:
                raise Exception('If the model is to be defined from scratch, input and output need to be defined.')

            self.init_sequential(n_input, n_output, **sequential_kwargs)

        elif issubclass(type(model), keras.Model):
            self.model = model

        else:
            raise TypeException('To import an external model as DenseKeras, it must be a subclass of keras.Model.')

        callbacks = []
        if early_stopping: 
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=10,
                restore_best_weights=True,
                verbose=1)
            callbacks.append(early_stopping_callback)

        if reduce_LR_plateau:
            reduce_LR_plateau_callback = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=5,
                factor=0.2,
                verbose=1)
            callbacks.append(reduce_LR_plateau_callback)

        if not len(callbacks) == 0:
            self.callbacks = callbacks

    def set_data_type(self, data_type):
        self.data_type = tf.Variable(data_type)

    def init_sequential(
        self, 
        n_input, 
        n_output, 
        hidden_layers=[64, 64],
        learning_rate=0.01,
        momentum=0.9,
        loss_function='categorical_crossentropy',
        dropout=0.4,
        discretization=False,
        l2_reg_input=False):

        self.model = keras.models.Sequential()
        activation_function = lambda x: keras.activations.relu(x, alpha=0.1)
        layers = [n_input] + hidden_layers + [n_output]
        layers_in_out = list(zip(layers[:-1], layers[1:]))
        for idx, layer in enumerate(layers_in_out):
            if idx == len(layers_in_out) - 1: # final layer
                activation = 'softmax'
                dropout_layer = 0.0

            else:
                activation = activation_function
                dropout_layer = dropout

            if idx == 0:
                self.model.add(keras.Input(
                    shape=(layer[0], )))
                if discretization:
                    self.model.add(keras.layers.Discretization(bin_boundaries=[-0.675, 0, 0.675]))

            if l2_reg_input:
                self.model.add(keras.layers.Dense(
                    input_shape=(layer[0], ),
                    units=layer[1],
                    kernel_initializer='glorot_uniform',                                          
                    bias_initializer='zeros',
                    activation=activation,
                    kernel_regularizer=keras.regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=keras.regularizers.L2(1e-4),
                    activity_regularizer=keras.regularizers.L2(1e-5)))

            else:
                self.model.add(keras.layers.Dense(
                    input_shape=(layer[0], ),
                    units=layer[1],
                    kernel_initializer='glorot_uniform',                                          
                    bias_initializer='zeros',
                    activation=activation))

            if dropout_layer > 0.0:
                self.model.add(keras.layers.Dropout(dropout_layer))

        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate, 
            momentum=momentum)
        self.compile(
            optimizer=optimizer, 
            loss=loss_function,
            metrics=['accuracy'])

    def reset_output(self, n_output):
        input_shape = (self.model.layers[-1].input.shape[1], )
        activation = self.model.layers[-1].activation
        self.model.pop()
        self.model.add(keras.layers.Dense(
                        input_shape=input_shape,
                        units=n_output,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        activation=activation)) 

    def call(self, inputs):
        return self.model.call(inputs)

    @classmethod
    def load_model(cls, path):
        # Reinitialize because simply using keras.load_model messes up the variable type of the model
        loaded_model = keras.models.load_model(path)
        model = cls(model=loaded_model.model)
        model.set_data_type(loaded_model.data_type)

        return model

    @classmethod
    def import_external(cls, model, feature_names, data_type, label_encoding, label_decoding, is_binary=False):
        if not issubclass(type(model), keras.Model):
            raise TypeException('To import an external model as DenseKeras, it must be a subclass of keras.Model.')

        pass

class DenseTorch(torch.nn.Module, DenseBase):
    """
    """

    def __init__(self, n_input, n_output, **kwargs):
        pass

    @classmethod
    def import_external(cls, model, feature_names, data_type, label_encoding, label_decoding, is_binary=False):
        if not issubclass(type(model), torch.nn.Module):
            raise TypeException('To import an external model as DenseTorch, it must be a subclass of torch.nn.Module.')

        pass