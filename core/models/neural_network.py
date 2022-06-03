import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

class NeuralNetwork():
    """Add explanation.
    """

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
        batch_size=40
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

    def train(self, x, y_onehot, y_int, **kwargs):
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
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=self.batch_size,
            epochs = self.max_epochs, 
            verbose = 0,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping_callback, reduce_LR_plateau_callback])
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

        pass

    def predict(self, input_vec):
        """Calculate and return label prediction of trained model for an input
        vector input_vec (dtype=int)"""

        # Returns only absolute decisions, nothing known about the difference 
        # in prediction confidence 
        pred_vec = np.argmax(self.model.predict(input_vec), axis = -1)

        return pred_vec

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
      
    def predict_proba(self, input_vec):
        """Calculate and return probabilities of label predictions of trained model for an input
        vector input_vec (dtype=int)"""

        proba_pred_vec = self.model.predict(input_vec)

        return proba_pred_vec